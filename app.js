let session = null;
let currentFile = null;
let desiredWidth = 0;
let pixMask = null;
let offscreenCanvas = document.createElement("canvas");
let filePrepared = false;
let scaleCorrect = false;
let downloadReady = false;
let downloaded = false;
let defaultWarningText = "<p>You should consider downloading <a href=\"https://chainner.app/\">chaiNNer</a> to run the model offline.</p>";
let defaultOverlayText = "<p>Drag and drop an image here to upload</p><p>or</p><p>Click to select an image from your computer</p>";


const maxTileSize = 392; // size of the tiles to process
const overlap = 8; // overlap between tiles

function setDownloadButtonsDisabledTo(disabled) {
    document.getElementById("download-1-button").disabled = disabled;
    document.getElementById("download-4-button").disabled = disabled;
    document.getElementById("download-d-button").disabled = disabled;
}

async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function init() {
    const textOverlay = document.getElementById("text-overlay")

    // fetch the pixMask image
    if (!pixMask) {
        textOverlay.innerHTML = "<p>Downloading pixel mask...</p>"
        let response = await fetch("./pixMask.png");
        if (!response.ok) {
            console.error("Failed to fetch pixMask.png");
            return;
        }
        let blob = await response.blob();
        pixMask = await createImageBitmap(blob);
    }

    // Initialize the session
    textOverlay.innerHTML = "<p>Loading model...</p>";
    const select = document.getElementById("model-dropdown");
    const selectedModel = select.options[select.selectedIndex].value;
    const eps = ["webnn", "webgpu", "wasm", "cpu"]

    for (const ep of eps) {
        try {
            session = await ort.InferenceSession.create(`./models/${selectedModel}`, {
                executionProviders: [ep],
                graphOptimizationLevel: "all",
            });
            console.debug(`Session created with ${ep}`);
            if (ep !== "webnn" && ep !== "webgpu") {
                defaultWarningText = "<p>WebGPU is not supported in your browser. Please download <a href=\"https://chainner.app/\">chaiNNer</a> to run the model offline.</p>";
                const warningArea = document.getElementById("warning-area");
                warningArea.innerHTML = defaultWarningText;
            }
            textOverlay.innerHTML = defaultOverlayText;
            break
        } catch (e) {
            console.error(`Failed to create session with ${ep}: ${e}`);
        }
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
    document.getElementById("text-overlay").innerHTML = ""
    defaultOverlayText = ""

    if (downloadReady && !downloaded) {
        const warningArea = document.getElementById("warning-area");
        warningArea.innerHTML = "<p>Please download the image before running another image.</p>";
        return;
    } else {
        const warningArea = document.getElementById("warning-area");
        warningArea.innerHTML = defaultWarningText;
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
    const scaleRange = document.getElementById("scale-range");
    const scaleNumber = document.getElementById("scale-number");
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

    const previewCanvas = document.getElementById("main-canvas")
    previewCanvas.width = offscreenCanvas.width;
    previewCanvas.height = offscreenCanvas.height;
    const pCtx = previewCanvas.getContext("2d");
    // turn off the smoothing
    pCtx.imageSmoothingEnabled = false;
    pCtx.drawImage(outputScaleCanvas, 0, 0, outputScaleCanvas.width, outputScaleCanvas.height, 0, 0, previewCanvas.width, previewCanvas.height);
    filePrepared = true;
}

async function runCurrentFile() {
    if (!filePrepared) {
        console.error("File not prepared");
        return;
    }

    const ctx = document.getElementById("main-canvas").getContext("2d");

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
            let output = await runOneTile(tile);
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

    downloadReady = true;
    setDownloadButtonsDisabledTo(false);
}


document.addEventListener("DOMContentLoaded", async function () {
    document.getElementById("run-button").addEventListener("click", async function () {
        await runCurrentFile();
    })
    const fileInput = document.getElementById("file-input")
    const canvasContainer = document.getElementById('canvas-container');
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
        if (files.length > 0) {
            currentFile = files[0];
            filePrepared = false;
            scaleCorrect = false;
            downloadReady = false;
            await prepareCurrentFile()
        }
    });
    canvasContainer.addEventListener('click', function (_) {
        if (downloadReady && !downloaded) {
            const warningArea = document.getElementById("warning-area");
            warningArea.innerHTML = "<p>Please download the image before running another image.</p>";
            return;
        } else {
            const warningArea = document.getElementById("warning-area");
            warningArea.innerHTML = defaultWarningText;
        }
        fileInput.click();
    });
    fileInput.addEventListener("change", async function (event) {
        const files = event.target.files;
        if (files.length > 0) {
            currentFile = files[0];
            filePrepared = false;
            scaleCorrect = false;
            await prepareCurrentFile()
        }
    })
    const scaleRange = document.getElementById("scale-range")
    const scaleNumber = document.getElementById("scale-number")
    scaleRange.addEventListener("change", async function (_) {
        scaleNumber.value = scaleRange.value;
        await prepareCurrentFile()  // TODO: debounce this
    })
    scaleNumber.addEventListener("change", async function (_) {
        scaleRange.value = scaleNumber.value;
        await prepareCurrentFile()  // TODO: debounce this
    })

    const download1Button = document.getElementById("download-1-button");
    const download4Button = document.getElementById("download-4-button");
    const downloadDButton = document.getElementById("download-d-button");
    setDownloadButtonsDisabledTo(true);
    download4Button.addEventListener("click", async function () {
        if (!downloadReady) return;
        const link = document.createElement("a");
        link.download = currentFile.name.replace(/\.[^/.]+$/, "") + "_4x_pixelated.png";
        link.href = document.getElementById("main-canvas").toDataURL();
        link.click();
        downloaded = true;
    });
    download1Button.addEventListener("click", async function () {
        if (!downloadReady) return;

        const mainCanvas = document.getElementById("main-canvas");
        const downscaleCanvas = document.createElement("canvas");
        downscaleCanvas.width = offscreenCanvas.width / 4;
        downscaleCanvas.height = offscreenCanvas.height / 4;
        const downscaleCtx = downscaleCanvas.getContext("2d");
        downscaleCtx.drawImage(mainCanvas, 0, 0, mainCanvas.width, mainCanvas.height, 0, 0, downscaleCanvas.width, downscaleCanvas.height);

        const link = document.createElement("a");
        link.download = currentFile.name.replace(/\.[^/.]+$/, "") + "_pixelated.png";
        link.href = downscaleCanvas.toDataURL();
        link.click();
        downloaded = true;
    });
    downloadDButton.addEventListener("click", async function () {
        if (!downloadReady) return;
        const canvas = document.getElementById("main-canvas")
        const ctx = canvas.getContext("2d");
        ctx.font = "30px Arial";
        ctx.fillStyle = "red";
        ctx.fillText("Discarded", 10, 50);
        downloaded = true;
    });

    document.getElementById("model-dropdown").addEventListener("change", async function (_) {
        await init();
    })

    await init()
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