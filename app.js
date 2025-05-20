let session = null;
let currentFile = null;
let desiredWidth = 0;
let pixMask = null;

const maxTileSize = 256; // size of the tiles to process
const overlap = 4; // overlap between tiles

async function init() {
    // fetch the pixMask image
    let response = await fetch("/pixMask.png");
    if (!response.ok) {
        console.error("Failed to fetch pixMask.png");
        return;
    }
    let blob = await response.blob();
    pixMask = await createImageBitmap(blob);

    // Initialize the session
    session = await ort.InferenceSession.create("models/1x_pixelate4x-v5-340k_G.onnx", {
        executionProviders: ["webgpu", "wasm", "webgl"],
        graphOptimizationLevel: "all",
    });
    console.debug("Session initialized");
    console.debug(session);
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
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(data, x, y, width, height, 0, 0, width, height);
    ctx.drawImage(pixMask, 0, 0, width, height, 0, 0, width, height);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

async function runCurrentFile() {
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

    // read contents of the image and add new <img> element to the page
    const bitmap = await createImageBitmap(currentFile);
    // TODO: get this from slider
    desiredWidth = Math.round(bitmap.width / 4);
    let desiredHeight = Math.round(desiredWidth / bitmap.width * bitmap.height);

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = desiredWidth * 4;
    canvas.height = desiredHeight * 4;
    ctx.drawImage(bitmap, 0, 0, bitmap.width, bitmap.height, 0, 0, canvas.width, canvas.height);

    let horizontalCount = 1 + Math.ceil((canvas.width - maxTileSize) / (maxTileSize - overlap));
    if (canvas.width <= maxTileSize) horizontalCount = 1;
    let verticalCount = 1 + Math.ceil((canvas.height - maxTileSize) / (maxTileSize - overlap));
    if (canvas.height <= maxTileSize) verticalCount = 1;
    let tileWidth = Math.ceil(canvas.width / horizontalCount / 4) * 4 + overlap;
    if (horizontalCount === 1) tileWidth = canvas.width;
    let tileHeight = Math.ceil(canvas.height / verticalCount / 4) * 4 + overlap;
    if (verticalCount === 1) tileHeight = canvas.height;

    const rv = document.createElement("canvas")
    rv.width = canvas.width;
    rv.height = canvas.height;
    const rctx = rv.getContext("2d");
    rctx.drawImage(canvas, 0, 0);

    document.body.appendChild(rv);
    console.debug(horizontalCount, verticalCount);
    for (let j = 0; j < verticalCount; j++) {
        for (let i = 0; i < horizontalCount; i++) {
            let x1 = Math.floor(i * (tileWidth - overlap));
            let y1 = Math.floor(j * (tileHeight - overlap));
            let x2 = Math.min(x1 + tileWidth, canvas.width);
            let y2 = Math.min(y1 + tileHeight, canvas.height);
            let width = x2 - x1;
            let height = y2 - y1;
            let tile = convertToTile(canvas, x1, y1, width, height);
            console.debug(`Tile x${i}y${j}: `, tile);
            // run the model
            let output = await runOneTile(tile);
            console.debug("Output: ", output);
            // convert to ImageData
            let imageData = output.toImageData()
            let imageBitmap = await createImageBitmap(imageData);
            console.debug("Converted to ImageBitmap: ", imageBitmap);
            // draw the output on the canvas
            let xOffset = i === 0 ? 0 : overlap;
            let yOffset = j === 0 ? 0 : overlap;
            rctx.drawImage(imageBitmap, xOffset, yOffset, width, height, x1 + xOffset, y1 + yOffset, width, height);
        }
    }
}


document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("imageInput").addEventListener("change", async function (event) {
        currentFile = event.target.files[0];
    });
    document.getElementById("runButton").addEventListener("click", async function () {
        await runCurrentFile();
    })
});

init().catch(console.error);

/* TODO:
[x] - Tiling (based on session memory)
[ ] - Nearest neighbor preview before running the model
[ ] - debounce the preview
[x] - Add a button to run the model
[ ] - Download button for the output image
[ ] - slider for the scale factor
[ ] - dropdown for the model
[ ] - more models
[ ] - drag and drop support
 */