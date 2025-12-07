/* global ort */
let pixelateSession = null;
let vertCombineSession = null;
let horiCombineSession = null;
let finalizerSession = null;
let currentImageBitmap = null;
let currentFileName = null;
let desiredSize = 0;
let outputCanvas = document.createElement("canvas");
let filePrepared = false;
let downloadReady = false;
let downloaded = false;
let running = false;
let estimatedTimePerPixel = 0;
let runningExecutionProvider = "";
let loadedPaletteTensors = {}
let selectedPalette = null;

// Get the elements from the DOM
const disclaimer = document.getElementById("disclaimer");
const textOverlay = document.getElementById("text-overlay");
let defaultOverlayText = "<p>Drag and drop an image here to upload</p><p>or</p><p>Click to select an image from your computer</p>";

const paletteDropdown = document.getElementById("palette-dropdown");
const scaleRange = document.getElementById("scale-range");
const scaleNumber = document.getElementById("scale-number");

const runButton = document.getElementById("run-button");
const download1Button = document.getElementById("download-1-button");
const download4Button = document.getElementById("download-4-button");
const downloadDButton = document.getElementById("download-d-button");

/** @type {HTMLCanvasElement} */
const mainCanvas = document.getElementById("main-canvas");
const fileInput = document.getElementById("file-input")
const paletteInput = document.getElementById("palette-input")
const canvasContainer = document.getElementById('canvas-container');

let maxTileSize = 256; // size of the tiles to process
let overlap = 64; // overlap between *input* tiles

function setDownloadButtonsDisabledTo(disabled) {
    download1Button.disabled = disabled;
    download4Button.disabled = disabled;
    downloadDButton.disabled = disabled;
}

async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function setTextOverlayInner(newHtml, updateVisibility = true) {
    if (newHtml && updateVisibility && textOverlay.hidden) {
        textOverlay.hidden = false;
        await sleep(50);
    } else if (newHtml.length === 0 && updateVisibility && !textOverlay.hidden) {
        textOverlay.hidden = true;
        await sleep(50);
    }
    textOverlay.innerHTML = newHtml;
    await sleep(1);
}

class TextOverlayPercentageScheduler {
    constructor() {
        this.running = false;
        this.current = 0;
        this.speed = 0; // how long it takes for the percentage to increment
    }

    /**
     * @param {number} currentPercentage - The current percentage
     * @param {number} speed - how long it takes for the percentage to increment
     */
    async start(currentPercentage, speed) {
        this.current = currentPercentage;
        this.speed = speed;
        textOverlay.hidden = false;

        if (runningExecutionProvider.includes('cpu')) {
            // wasm and cpu will block the event loop, no need to start the scheduler
            return;
        }

        while (this.speed === 0) {
            await setTextOverlayInner(`<p>${this.current}%</p>`)
            await sleep(1000)
        }

        this.running = true;
        console.log(`Starting percentage scheduler with speed ${this.speed}`);
        await setTextOverlayInner(`<p>${this.current}%</p>`, false);

        while (this.current < 100) {
            await sleep(this.speed);
            this.current += 1;
            await setTextOverlayInner(`<p>${this.current}%</p>`, false);
            if (!this.running) {
                await setTextOverlayInner("");
                return;
            }
        }
    }

    /**
     * @param {number} currentPercentage - The current percentage
     * @param {number} speed - how long it takes for the percentage to increment
     */
    async update(currentPercentage, speed) {
        console.debug(`Updating percentage: ${currentPercentage}, Speed: ${speed}`);
        this.current = currentPercentage;
        this.speed = speed;
        if (!this.running) {
            await setTextOverlayInner(`<p>${currentPercentage}%</p>`, false);
        }
    }

    stop() {
        this.running = false;
        setTextOverlayInner("").catch(console.error);
    }
}

async function runCurrentFile() {
    if (await checkProceedingWithFileDisabled()) return;
    if (!filePrepared) {
        await setTextOverlayInner(`
<p>Please upload an image first</p>
<p>Drag and drop an image here to upload</p>
<p>or</p>
<p>Click to select an image from your computer</p>
`);
        runButton.disabled = true;
        return;
    }
    if (!selectedPalette) {
        await setTextOverlayInner("<p>Please select a palette from the bottom-left dropdown first</p>");
        runButton.disabled = true;
        return;
    }
    if (!finalizerSession) {
        await setTextOverlayInner("<p>Wait for the model to finish loading</p>");
        return
    }

    running = true;
    downloadReady = false;
    downloaded = false;
    setDownloadButtonsDisabledTo(true);
    await setTextOverlayInner('Processing image...');

    const maxSize = scaleRange.max;
    let maxPossiblePower = Math.min(Math.floor(Math.log2(maxSize / desiredSize)), 3)
    console.log(`Max possible power: ${maxPossiblePower}`);
    let scale = Math.pow(2, maxPossiblePower);

    let desiredWidth = Math.round(currentImageBitmap.width * desiredSize / maxSize)
    let desiredHeight = Math.round(currentImageBitmap.height * desiredSize / maxSize)
    let inputWidth = desiredWidth * scale;
    let inputHeight = desiredHeight * scale;

    let inputCanvas = document.createElement('canvas');
    inputCanvas.width = inputWidth;
    inputCanvas.height = inputHeight;
    let inputCtx = inputCanvas.getContext('2d');
    inputCtx.drawImage(currentImageBitmap, 0, 0, inputWidth, inputHeight);

    // tiling
    let previewWScale = currentImageBitmap.width / inputWidth;
    let previewHScale = currentImageBitmap.height / inputHeight;
    mainCanvas.width = currentImageBitmap.width;
    mainCanvas.height = currentImageBitmap.height;
    let ctx = mainCanvas.getContext("2d");
    // draw the image, then disable image smoothing
    ctx.drawImage(
        inputCanvas,
        0, 0, inputWidth, inputHeight,
        0, 0, currentImageBitmap.width, currentImageBitmap.height
    );
    ctx.imageSmoothingEnabled = false;

    const percentageScheduler = new TextOverlayPercentageScheduler();
    let totalArea = inputWidth * inputHeight;
    let areaRemaining = totalArea;
    let speed = totalArea * estimatedTimePerPixel / 100;

    // deliberately sync call, do not await as it will block all the code after from running
    percentageScheduler.start(0, speed).catch(console.error);


    let tileSize = Math.floor(maxTileSize / scale) * scale;
    let tileOverlap = Math.floor(overlap / scale) * scale;
    let halfTileSize = Math.floor(tileSize / scale / 2) * scale;

    let heightRemaining = inputHeight;
    let heightCombined = null
    while (heightRemaining > 0) {
        let widthRemaining = inputWidth;
        let widthCombined = null

        let tileHeight = Math.min(heightRemaining, tileSize);
        let lastVTile = (heightRemaining === tileHeight);
        if (heightRemaining - tileHeight + tileOverlap < halfTileSize && !lastVTile)
            tileHeight = halfTileSize;


        while (widthRemaining > 0) {
            let startTime = Date.now();

            let tileWidth = Math.min(widthRemaining, tileSize);
            let lastHTile = (widthRemaining === tileWidth);
            if (widthRemaining - tileWidth + tileOverlap < halfTileSize && !lastHTile)
                tileWidth = halfTileSize;
            let tileX = inputWidth - widthRemaining;
            let tileY = inputHeight - heightRemaining;


            let tileCanvas = document.createElement('canvas');
            tileCanvas.width = Math.min(tileWidth, widthRemaining + tileOverlap);
            tileCanvas.height = Math.min(tileHeight, heightRemaining + tileOverlap);
            let tileCtx = tileCanvas.getContext('2d');
            tileCtx.drawImage(inputCanvas, tileX, tileY, tileWidth, tileHeight, 0, 0, tileWidth, tileHeight);

            console.log(`Processing tile at (${tileX}, ${tileY}) with size ${tileWidth}x${tileHeight}`);
            let tileTensor = await ort.Tensor.fromImage(tileCtx.getImageData(0, 0, tileCanvas.width, tileCanvas.height))
            let tileOutput = await pixelateSession.run({
                'image': tileTensor,
                'palette': loadedPaletteTensors[selectedPalette],
                'downscale_power': new ort.Tensor('int64', new BigInt64Array([BigInt(maxPossiblePower)]), [])
            });
            tileTensor.dispose();
            let endTime = Date.now();
            let elapsedTime = endTime - startTime;
            let usefulWidth = tileWidth - (lastHTile ? 0 : tileOverlap);
            let usefulHeight = tileHeight - (lastVTile ? 0 : tileOverlap);
            let thisTimePerPixel = elapsedTime / (usefulWidth * usefulHeight);
            areaRemaining -= usefulWidth * usefulHeight;

            if (tileX === 0 && tileY === 0) {
                // skip the very first tile, model initialization time is long
            } else {
                if (estimatedTimePerPixel === 0) {
                    estimatedTimePerPixel = thisTimePerPixel;
                } else {
                    estimatedTimePerPixel = (estimatedTimePerPixel + thisTimePerPixel) / 2;
                }
            }

            let newSpeed = totalArea * estimatedTimePerPixel / 100;
            let currentProgress = Math.round(100 * (totalArea - areaRemaining) / totalArea);
            await percentageScheduler.update(currentProgress, newSpeed);

            let finalOutput = await finalizerSession.run({
                'probs': tileOutput['out_probs'],
                'palette': loadedPaletteTensors[selectedPalette],
            });

            let imageBitmap = await createImageBitmap(finalOutput['image'].toImageData());
            ctx.drawImage(
                imageBitmap,
                0, 0, tileWidth / scale, tileHeight / scale,
                tileX * previewWScale, tileY * previewHScale, tileWidth * previewWScale, tileHeight * previewHScale
            );

            if (widthCombined === null)
                widthCombined = tileOutput['out_probs']
            else {
                console.log("Combining horizontal tiles")
                let combinedTensor = await horiCombineSession.run({
                    'a': widthCombined,
                    'b': tileOutput['out_probs'],
                    'overlap': new ort.Tensor('int64', new BigInt64Array([BigInt(tileOverlap / scale)]), [1])
                })
                widthCombined = combinedTensor['out']
            }

            widthRemaining -= tileWidth - tileOverlap
            console.log(`widthLeft: ${widthRemaining}`)
            if (lastHTile) break;
        }
        if (heightCombined === null)
            // noinspection JSSuspiciousNameCombination
            heightCombined = widthCombined
        else {
            console.log("Combining vertical tiles")
            let combinedTensor = await vertCombineSession.run({
                'a': heightCombined,
                'b': widthCombined,
                'overlap': new ort.Tensor('int64', new BigInt64Array([BigInt(tileOverlap / scale)]), [1])
            })
            heightCombined = combinedTensor['out']
        }
        heightRemaining -= tileHeight - tileOverlap
        if (lastVTile) break;
    }
    let finalOutput = await finalizerSession.run({
        'probs': heightCombined,
        'palette': loadedPaletteTensors[selectedPalette],
    });
    let imageData = finalOutput['image'].toImageData();
    console.log(imageData)
    let imageBitmap = await createImageBitmap(imageData);
    ctx.clearRect(0, 0, currentImageBitmap.width, currentImageBitmap.height);
    ctx.drawImage(
        imageBitmap,
        0, 0, desiredWidth, desiredHeight,
        0, 0, currentImageBitmap.width, currentImageBitmap.height
    );

    outputCanvas.width = desiredWidth;
    outputCanvas.height = desiredHeight;
    let outputCtx = outputCanvas.getContext('2d');
    outputCtx.drawImage(imageBitmap, 0, 0);
    await setTextOverlayInner("");
    downloadReady = true;
    setDownloadButtonsDisabledTo(false);
    running = false;
    percentageScheduler.stop();
}

async function checkProceedingWithFileDisabled() {
    if (running) {
        return true;
    }
    if (downloadReady && !downloaded) {
        await setTextOverlayInner("<p>Please download the image before running another image.</p>")
        return true;
    }
    return false;
}

async function handleFileUpload(file) {
    if (await checkProceedingWithFileDisabled()) return;

    downloaded = false;
    downloadReady = false;
    filePrepared = false;
    setDownloadButtonsDisabledTo(true);
    await setTextOverlayInner('Loading image...');

    currentImageBitmap = await createImageBitmap(file);
    currentFileName = file.name;
    let maxSize = Math.max(currentImageBitmap.width, currentImageBitmap.height);
    scaleRange.max = scaleNumber.max = maxSize;
    scaleRange.value = scaleNumber.value = Math.min(300, maxSize / 2);
    await setTextOverlayInner(selectedPalette != null ? "" : "<p>Select a palette from the bottom-left dropdown</p>");
    defaultOverlayText = ""
    await drawCurrentImage();
    filePrepared = true;
    if (selectedPalette)
        runButton.disabled = false;
}

async function drawCurrentImage() {
    if (!currentImageBitmap) return;

    mainCanvas.width = currentImageBitmap.width;
    mainCanvas.height = currentImageBitmap.height;
    desiredSize = scaleRange.value;
    const maxSize = scaleRange.max;

    let scalingCanvas = document.createElement('canvas')
    let sCtx = scalingCanvas.getContext('2d')
    scalingCanvas.width = Math.round(currentImageBitmap.width * desiredSize / maxSize);
    scalingCanvas.height = Math.round(currentImageBitmap.height * desiredSize / maxSize);
    sCtx.drawImage(currentImageBitmap, 0, 0, scalingCanvas.width, scalingCanvas.height);

    let ctx = mainCanvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(scalingCanvas, 0, 0, scalingCanvas.width, scalingCanvas.height, 0, 0, mainCanvas.width, mainCanvas.height);
}

/**
 * @param {Blob} userPalette - optional user-provided palette blob
 * @param {string} paletteName - name of the palette to load
 */
async function loadOrFetchPalette(userPalette = null, paletteName = null) {
    if (!userPalette && !paletteName) {
        console.error("loadOrFetchPalette called without a palette");
        return null;
    }

    if (!userPalette && paletteName) {
        let cachedTensor = loadedPaletteTensors[paletteName]
        if (cachedTensor) {
            selectedPalette = paletteName;
            return;
        }
    }

    // use the most optimal way to get the palette blob
    let paletteBlob = userPalette
    if (!paletteBlob) {
        let paletteResponse = await fetch(`./palettes/${paletteName}.png`);
        if (!paletteResponse.ok) {
            console.error(`Failed to fetch palette ${paletteResponse.url} - ${paletteResponse.statusText}`);
            return null;
        }
        paletteBlob = await paletteResponse.blob();
    }

    // create canvas to later get ImageData from
    const imageBitmap = await createImageBitmap(paletteBlob);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = imageBitmap.width;
    canvas.height = imageBitmap.height;
    ctx.drawImage(imageBitmap, 0, 0);

    // convert to Tensor and reshape to (B, C, N)
    const tensor = await ort.Tensor.fromImage(ctx.getImageData(0, 0, canvas.width, canvas.height))
    console.log(`Loaded palette ${paletteName} with shape ${tensor.dims}`);
    await setTextOverlayInner(defaultOverlayText)
    loadedPaletteTensors[paletteName] = tensor.reshape([1, 3, canvas.height * canvas.width]);
    selectedPalette = paletteName;
    if (filePrepared) {
        runButton.disabled = false
    }
}

async function initializeModel(url, eps = [["webnn", "webgpu"], ["wasm", "cpu"]]) {
    let rv = null;

    const modelResponse = await fetch(url, {'cache': 'force-cache'});
    if (!modelResponse.ok) {
        console.error(`Failed to fetch model`);
        await setTextOverlayInner("<p>Failed to download model</p>");
        return;
    }
    const modelBuffer = await modelResponse.arrayBuffer();

    let shouldRunOnGPU = false

    for (const ep of eps) {
        if (ep.includes("webgpu")) shouldRunOnGPU = true;
        try {
            rv = await ort.InferenceSession.create(modelBuffer, {
                executionProviders: [...ep],
                graphOptimizationLevel: "all",
            });
            console.debug(`Session created with ${ep}`);
            runningExecutionProvider = ep.join(", ");
            if (!ep.includes("webgpu") && shouldRunOnGPU) {
                disclaimer.innerHTML = "<p>⚠️ WebGPU failed. Model is running on CPU - this can cause performance issues.</p>"
            }
            break
        } catch (e) {
            console.error(`Failed to create session with ${ep}: ${e}`);
        }
    }

    return rv
}

async function initializeAllModels() {
    pixelateSession = await initializeModel('./onnx/model202512072323.onnx')
    vertCombineSession = await initializeModel('./onnx/vertical_overlap.onnx')
    horiCombineSession = await initializeModel('./onnx/horizontal_overlap.onnx')
    // finalizer has to live on CPU, otherwise switching palettes doesn't work
    finalizerSession = await initializeModel('./onnx/image_finalizer.onnx', [["wasm", "cpu"]])
    runButton.innerText = "✨ Pixelate ✨"
    await setTextOverlayInner(defaultOverlayText)
}

document.addEventListener("DOMContentLoaded", async function () {
    setDownloadButtonsDisabledTo(true);
    runButton.disabled = true;
    // prevent leaving the page if the model is running
    window.addEventListener('beforeunload', function (event) {
        if (running || downloadReady && !downloaded) {
            event.preventDefault();
        }
    })


    // button events
    runButton.addEventListener("click", async function () {
        await runCurrentFile();
    })
    scaleRange.addEventListener("input", async function (_) {
        scaleNumber.value = scaleRange.value;
        await drawCurrentImage()  // TODO: debounce this
    })
    scaleNumber.addEventListener("change", async function (_) {
        scaleRange.value = scaleNumber.value;
        await drawCurrentImage()  // TODO: debounce this
    })
    download1Button.addEventListener("click", async function () {
        if (!downloadReady) return;
        const selectedPalette = paletteDropdown.options[paletteDropdown.selectedIndex].value;
        const link = document.createElement("a");
        link.download = `${currentFileName.replace(/\.[^/.]+$/, "")}_1x_${selectedPalette}.png`;
        link.href = outputCanvas.toDataURL();
        link.click();
        downloaded = true;
    });
    download4Button.addEventListener("click", async function () {
        if (!downloadReady) return;
        const selectedPalette = paletteDropdown.options[paletteDropdown.selectedIndex].value;
        const link = document.createElement("a");
        link.download = `${currentFileName.replace(/\.[^/.]+$/, "")}_4x_${selectedPalette}.png`;
        let canvas = document.createElement("canvas");
        canvas.width = outputCanvas.width * 4;
        canvas.height = outputCanvas.height * 4;
        let ctx = canvas.getContext("2d");
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(outputCanvas, 0, 0, outputCanvas.width, outputCanvas.height, 0, 0, canvas.width, canvas.height);
        link.href = canvas.toDataURL();
        link.click();
        downloaded = true;
    })
    downloadDButton.addEventListener("click", async function () {
        if (!downloadReady) return;
        await setTextOverlayInner(`
<p>This image has been discarded</p>
<p>You can now change the image or run another model</p>
<p>Right now the image can still be downloaded</p>
`);
        downloaded = true;
    });


    // file input via drag and drop
    canvasContainer.addEventListener('dragover', function (event) {
        event.preventDefault();
        canvasContainer.classList.add('dragover');
        event.dataTransfer.dropEffect = 'copy';
    });
    canvasContainer.addEventListener('dragleave', function (_) {
        canvasContainer.classList.remove('dragover');
    });
    canvasContainer.addEventListener('drop', async function (event) {
        event.preventDefault();
        canvasContainer.classList.remove('dragover');
        const files = event.dataTransfer.files;
        await handleFileUpload(files[0]);
    });

    // file input via click
    canvasContainer.addEventListener('click', async function (_) {
        if (await checkProceedingWithFileDisabled()) return;
        fileInput.click();
    });
    fileInput.addEventListener("change", async function (event) {
        const files = event.target.files;
        await handleFileUpload(files[0]);
    })

    // palette input via click
    paletteInput.addEventListener("change", async function (event) {
        const files = event.target.files;
        let name = files[0].name.replace(".png", "");
        let existingOption = paletteDropdown.querySelector(`option[value="${name}"]`);
        if (existingOption) {
            existingOption.selected = true;
        } else {
            let newOption = document.createElement("option");
            newOption.value = name;
            newOption.textContent = name;
            paletteDropdown.appendChild(newOption);
            newOption.selected = true;
        }

        await loadOrFetchPalette(files[0], name);
    })
    // palette input via dropdown
    paletteDropdown.addEventListener("change", async function (_) {
        let currentURL = new URL(window.location.href);
        let selectedPalette = paletteDropdown.options[paletteDropdown.selectedIndex];
        currentURL.searchParams.set("palette", selectedPalette.value);
        history.pushState({}, '', currentURL);
        await loadOrFetchPalette(null, selectedPalette.value);
    })

    // functions to run after load
    // palette input via URL parameter
    let paletteFromParams = new URLSearchParams(window.location.search).get("palette");
    if (paletteFromParams) {
        // select that palette from the dropdown
        console.log(`Loading palette ${paletteFromParams} from URL params`);
        let paletteOption = paletteDropdown.querySelector(`option[value="${paletteFromParams}"]`);
        if (paletteOption) {
            paletteOption.selected = true;
            await loadOrFetchPalette(null, paletteOption.value);
        } else {
            console.error(`Palette ${paletteFromParams} not found`);
        }
    }
    // model initialization
    await initializeAllModels();
})


/* TODO:
[x] - Tiling (based on session memory)
[x] - Nearest neighbor preview before running the model
[ ] - real-time preview & debounce the preview
[x] - allow upload of palettes
[x] - Add a button to run the model
[x] - Download button for the output image
[x] - slider for the scale factor
[x] - dropdown for the palette
[x] - drag and drop support
[x] - Add loading indicator during model processing
[ ] - Implement error handling for failed model runs
[x] - Add image size validation and warnings
[ ] - Add keyboard shortcuts for common actions
[ ] - Save user preferences in local storage
[.] - Create shareable URLs for specific settings
 */