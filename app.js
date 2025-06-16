/* global ort */
let session = null;
let currentFile = null;
let desiredWidth = 0;
let pixMask = null;
let offscreenCanvas = document.createElement("canvas");
let filePrepared = false;
let scaleCorrect = false;
let downloadReady = false;
let downloaded = false;
let running = false;
let alreadyDownloadedModels = {}
let estimatedTimePerPixel = 0;

// Get the elements from the DOM
const disclaimer = document.getElementById("disclaimer");
const textOverlay = document.getElementById("text-overlay");
let defaultOverlayText = "<p>Drag and drop an image here to upload</p><p>or</p><p>Click to select an image from your computer</p>";

const modelDropdown = document.getElementById("model-dropdown");
const scaleRange = document.getElementById("scale-range");
const scaleNumber = document.getElementById("scale-number");

const runButton = document.getElementById("run-button");
const download1Button = document.getElementById("download-1-button");
const download4Button = document.getElementById("download-4-button");
const downloadDButton = document.getElementById("download-d-button");

/** @type {HTMLCanvasElement} */
const mainCanvas = document.getElementById("main-canvas");
const fileInput = document.getElementById("file-input")
const canvasContainer = document.getElementById('canvas-container');


let maxTileSize = 440; // size of the tiles to process
let overlap = 24; // overlap between tiles

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

    async start(currentPercentage, speed) {
        this.current = currentPercentage;
        this.speed = speed;
        await setTextOverlayInner(" ")  // this is to clear the hidden attribute

        while (this.speed === 0) {
            await setTextOverlayInner(`<p>${this.current}%</p>`)
            await sleep(1000)
        }

        this.running = true;
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

    async update(currentPercentage, speed) {
        console.debug(`Updating percentage: ${currentPercentage}, Speed: ${speed}`);
        this.current = currentPercentage;
        this.speed = speed;
        await setTextOverlayInner(`<p>${this.current}%</p>`, false);
    }

    stop() {
        this.running = false;
        setTextOverlayInner("").catch(console.error);
    }
}

async function init(userModel = null, modelName = null) {
    // fetch the pixMask image
    if (!pixMask) {
        await setTextOverlayInner("<p>Downloading pixel mask...</p>");
        let response = await fetch("./pixMask.png");
        if (!response.ok) {
            console.error("Failed to fetch pixMask.png");
            await setTextOverlayInner("<p>Failed to download pixel mask</p>");
            return;
        }
        let blob = await response.blob();
        pixMask = await createImageBitmap(blob);
    }

    // Initialize the session
    let modelBuffer = userModel;
    if (!modelBuffer) {
        await setTextOverlayInner("<p>Downloading model...</p>");
        const selectedModel = modelDropdown.options[modelDropdown.selectedIndex].value;


        modelBuffer = alreadyDownloadedModels[selectedModel];

        // fetch a buffer of the model (to allow browsers to cache it)
        if (!modelBuffer) {
            const modelURL = `./models/${selectedModel}`;
            const modelResponse = await fetch(modelURL, {'cache': 'force-cache'});
            if (!modelResponse.ok) {
                console.error(`Failed to fetch model from ${modelURL}`);
                await setTextOverlayInner("<p>Failed to download model</p>");
                return;
            }
            modelBuffer = await modelResponse.arrayBuffer();
            alreadyDownloadedModels[selectedModel] = modelBuffer;
        }
    } else {
        alreadyDownloadedModels[modelName] = userModel;
        const newOption = document.createElement("option");
        newOption.value = modelName;
        newOption.textContent = `Local file: ${modelName}`;
        newOption.selected = true;
        newOption.setAttribute('data-saveas', modelName);
        modelDropdown.appendChild(newOption);
    }

    const eps = [["webnn", "webgpu"], ["wasm"], ["cpu"]];

    await setTextOverlayInner("<p>Loading model...</p>");

    for (const ep of eps) {
        try {
            session = await ort.InferenceSession.create(modelBuffer, {
                executionProviders: [...ep],
                graphOptimizationLevel: "all",
            });
            console.debug(`Session created with ${ep}`);
            if (!ep.includes("webgpu")) {
                disclaimer.innerHTML = "<p>⚠️ WebGPU failed. Model is running on CPU - this can cause performance issues.</p>"
            }
            await setTextOverlayInner(defaultOverlayText);
            break
        } catch (e) {
            console.error(`Failed to create session with ${ep}: ${e}`);
        }
    }
    if (!session) {
        console.error("Failed to create session with any execution provider");
        await setTextOverlayInner("<p>Failed to load model</p>");
        return;
    }

    console.debug("Session initialized");
    console.debug(session);

    if (currentFile && !filePrepared) {
        await prepareCurrentFile();
    }
}

async function runOneTile(data) {
    // convert to Tensor
    let tensor = await ort.Tensor.fromImage(data)
    console.debug(tensor);
    // run the model
    console.debug("Running model...");
    const modelOutput = await session.run({"input": tensor});
    console.debug(modelOutput);
    return modelOutput["output"];
}

function convertToTile(data, x, y, width, height) {
    console.debug("Converting to tile: ", data, x, y, width, height);
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(data, x, y, width, height, 0, 0, width, height);
    ctx.drawImage(pixMask, 0, 0, width, height, 0, 0, width, height);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

async function prepareCurrentFile() {
    if (downloadReady && !downloaded) {
        await setTextOverlayInner("<p>Please download the image before running another image.</p>");
        return;
    } else {
        await setTextOverlayInner("");
        defaultOverlayText = ""
    }

    if (!currentFile) {
        console.error("No file selected");
        return;
    }

    if (!session) {
        console.error("Session not initialized");
        return;
    }

    if (!pixMask) {
        console.error("pixMask not initialized");
        return;
    }

    downloadReady = false;
    downloaded = false;
    setDownloadButtonsDisabledTo(true);

    // read contents of the image and add new <img> element to the page
    const bitmap = await createImageBitmap(currentFile);
    if (!scaleCorrect) {
        scaleRange.max = scaleNumber.max = Math.ceil(bitmap.width / 4);
        scaleRange.value = scaleNumber.value = Math.round(bitmap.width / 4)
        scaleRange.min = scaleNumber.min = 1;
        scaleRange.step = scaleNumber.step = 1;
        scaleCorrect = true;
    }
    desiredWidth = Math.round(scaleRange.value);
    let desiredHeight = Math.round(desiredWidth / bitmap.width * bitmap.height);

    const ctx = offscreenCanvas.getContext("2d");
    offscreenCanvas.width = desiredWidth * 4;
    offscreenCanvas.height = desiredHeight * 4;
    ctx.drawImage(bitmap, 0, 0, bitmap.width, bitmap.height, 0, 0, offscreenCanvas.width, offscreenCanvas.height);

    const outputScaleCanvas = document.createElement("canvas");
    outputScaleCanvas.width = offscreenCanvas.width / 4;
    outputScaleCanvas.height = offscreenCanvas.height / 4;
    const outputScaleCtx = outputScaleCanvas.getContext("2d");
    outputScaleCtx.drawImage(offscreenCanvas, 0, 0, offscreenCanvas.width, offscreenCanvas.height, 0, 0, outputScaleCanvas.width, outputScaleCanvas.height);

    const previewCanvas = mainCanvas
    previewCanvas.width = offscreenCanvas.width;
    previewCanvas.height = offscreenCanvas.height;
    const pCtx = previewCanvas.getContext("2d");
    // turn off the smoothing
    pCtx.imageSmoothingEnabled = false;
    pCtx.drawImage(outputScaleCanvas, 0, 0, outputScaleCanvas.width, outputScaleCanvas.height, 0, 0, previewCanvas.width, previewCanvas.height);
    filePrepared = true;
    runButton.disabled = false;
}

async function drawFileNotPixelated() {
    if (!currentFile) {
        console.error("No file selected");
        return;
    }
    const bitmap = await createImageBitmap(currentFile);
    const ctx = mainCanvas.getContext("2d");
    mainCanvas.width = bitmap.width;
    mainCanvas.height = bitmap.height;
    ctx.drawImage(bitmap, 0, 0, bitmap.width, bitmap.height, 0, 0, mainCanvas.width, mainCanvas.height);
}

async function runCurrentFile() {
    if (!filePrepared) {
        console.error("File not prepared");
        return;
    }

    if (downloadReady && !downloaded) {
        await setTextOverlayInner("<p>Please download the image before running another image.</p>");
        return;
    }

    if (running) {
        console.warn("Already running the model");
        return;
    }

    running = true;
    runButton.disabled = true;

    const ctx = mainCanvas.getContext("2d");

    overlap = Math.round(overlap / 4) * 4; // ensure overlap is a multiple of 4
    maxTileSize = Math.round(maxTileSize / 4) * 4; // ensure maxTileSize is a multiple of 4
    const doubleOverlap = overlap * 2;

    let horizontalCount = 1 + Math.ceil((offscreenCanvas.width - maxTileSize) / (maxTileSize - doubleOverlap));
    if (offscreenCanvas.width <= maxTileSize) horizontalCount = 1;
    let verticalCount = 1 + Math.ceil((offscreenCanvas.height - maxTileSize) / (maxTileSize - doubleOverlap));
    if (offscreenCanvas.height <= maxTileSize) verticalCount = 1;
    let tileWidth = Math.ceil(offscreenCanvas.width / horizontalCount / 4) * 4 + doubleOverlap;
    if (horizontalCount === 1) tileWidth = offscreenCanvas.width;
    let tileHeight = Math.ceil(offscreenCanvas.height / verticalCount / 4) * 4 + doubleOverlap;
    if (verticalCount === 1) tileHeight = offscreenCanvas.height;
    console.debug(horizontalCount, verticalCount);

    let totalCount = horizontalCount * verticalCount;
    let totalArea = tileWidth * tileHeight * totalCount;
    let doneCount = 0;
    let speed = totalArea / 100 * estimatedTimePerPixel;
    let percentageScheduler = new TextOverlayPercentageScheduler();
    if (totalCount !== 1 || speed !== 0) {
        // deliberately sync call, do not await as it will block all the code after from running
        percentageScheduler.start(0, speed).catch(console.error);
    } else {
        await setTextOverlayInner("<p>Pixelating...</p>");
    }

    let startTime = Date.now();
    for (let j = 0; j < verticalCount; j++) {
        for (let i = 0; i < horizontalCount; i++) {
            console.debug(`Tile x${i}y${j}`);
            let x1 = Math.floor(i * (tileWidth - doubleOverlap));
            let y1 = Math.floor(j * (tileHeight - doubleOverlap));
            let x2 = Math.min(x1 + tileWidth, offscreenCanvas.width);
            let y2 = Math.min(y1 + tileHeight, offscreenCanvas.height);
            let width = x2 - x1;
            let height = y2 - y1;

            let tile = convertToTile(offscreenCanvas, x1, y1, width, height);

            // run the model
            let startTime = Date.now();
            let output = await runOneTile(tile);
            doneCount++;
            let endTime = Date.now();
            let thisSpeed = (endTime - startTime) / (1 / totalCount * 100) + 10;

            speed = speed === 0 ? thisSpeed : (speed + thisSpeed) / 2;
            if (totalCount !== 1) {
                await percentageScheduler.update(Math.round(doneCount / totalCount * 100), speed);
            }

            let imageData = output.toImageData()
            let imageBitmap = await createImageBitmap(imageData);
            console.debug("Converted to ImageBitmap: ", imageBitmap);
            // draw the output on the canvas
            let xOffset = i === 0 ? 0 : overlap;
            let yOffset = j === 0 ? 0 : overlap;
            ctx.drawImage(imageBitmap, xOffset, yOffset, width, height, x1 + xOffset, y1 + yOffset, width, height);
            await sleep(10);
        }
    }
    let endTime = Date.now();

    if (estimatedTimePerPixel === 0) {
        estimatedTimePerPixel = (endTime - startTime) / totalArea;
    } else {
        let thisETA = (endTime - startTime) / totalArea;
        estimatedTimePerPixel = (estimatedTimePerPixel + thisETA) / 2;
    }

    percentageScheduler.stop()
    downloadReady = true;
    running = false;
    setDownloadButtonsDisabledTo(false);
    await setTextOverlayInner("");
}

/**
 * @param file {File} - The file that was uploaded
 * @returns {Promise<void>}
 */
async function handleFileUpload(file) {
    if (!file) {
        console.error("No file provided");
        return;
    }
    // handle image file upload
    if (file.type.startsWith("image/")) {
        currentFile = file;
        filePrepared = false;
        scaleCorrect = false;
        downloadReady = false;
        downloaded = false;

        if (!session) {
            await drawFileNotPixelated();
        } else {
            await prepareCurrentFile();
        }
    }
    // handle .onnx file upload
    else if (file.name.endsWith(".onnx")) {
        await init(await file.arrayBuffer(), file.name);
    }

}

document.addEventListener("DOMContentLoaded", async function () {
    runButton.addEventListener("click", async function () {
        await runCurrentFile();
    })

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
    canvasContainer.addEventListener('click', function (_) {
        if (running) return;
        if (downloadReady && !downloaded) {
            setTextOverlayInner("<p>Please download the image before running another image.</p>").catch(console.error);
            return;
        } else if (session) {
            setTextOverlayInner(defaultOverlayText).catch(console.error);
        }
        fileInput.click();
    });
    fileInput.addEventListener("change", async function (event) {
        const files = event.target.files;
        await handleFileUpload(files[0]);
    })
    scaleRange.addEventListener("change", async function (_) {
        scaleNumber.value = scaleRange.value;
        await prepareCurrentFile()  // TODO: debounce this
    })
    scaleNumber.addEventListener("change", async function (_) {
        scaleRange.value = scaleNumber.value;
        await prepareCurrentFile()  // TODO: debounce this
    })

    setDownloadButtonsDisabledTo(true);
    download4Button.addEventListener("click", async function () {
        if (!downloadReady) return;
        const selectedModel = modelDropdown.options[modelDropdown.selectedIndex].getAttribute('data-saveas');
        const link = document.createElement("a");
        link.download = `${currentFile.name.replace(/\.[^/.]+$/, "")}_4x_${selectedModel}.png`;
        link.href = mainCanvas.toDataURL();
        link.click();
        downloaded = true;
    });
    download1Button.addEventListener("click", async function () {
        if (!downloadReady) return;

        const downscaleCanvas = document.createElement("canvas");
        downscaleCanvas.width = offscreenCanvas.width / 4;
        downscaleCanvas.height = offscreenCanvas.height / 4;
        const downscaleCtx = downscaleCanvas.getContext("2d");
        downscaleCtx.drawImage(mainCanvas, 0, 0, mainCanvas.width, mainCanvas.height, 0, 0, downscaleCanvas.width, downscaleCanvas.height);

        const selectedModel = modelDropdown.options[modelDropdown.selectedIndex].getAttribute('data-saveas');
        const link = document.createElement("a");
        link.download = `${currentFile.name.replace(/\.[^/.]+$/, "")}_${selectedModel}.png`;
        link.href = downscaleCanvas.toDataURL();
        link.click();
        downloaded = true;
    });
    downloadDButton.addEventListener("click", async function () {
        if (!downloadReady) return;
        await setTextOverlayInner(`
<p>This image has been discarded</p>
<p>You can now change the image or run another model</p>
<p>Right now the image can still be downloaded</p>
`);
        downloaded = true;
    });

    modelDropdown.addEventListener("change", async function (_) {
        await init();
    })

    window.addEventListener('beforeunload', function (event) {
        if (running || downloadReady && !downloaded) {
            event.preventDefault();
        }
    })
});


/* TODO:
[x] - Tiling (based on session memory)
[x] - Nearest neighbor preview before running the model
[ ] - real-time preview & debounce the preview
[x] - Add a button to run the model
[x] - Download button for the output image
[x] - slider for the scale factor
[x] - dropdown for the model
[x] - more models
[x] - drag and drop support
 */