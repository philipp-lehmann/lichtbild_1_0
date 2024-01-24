let video, faceapi;
let boxes = [];
let boxId = 0;
let prevBoxes = [];
let currentFaceImg = null;
let currentEdgeImg = null;


currentFaceImg = null;

let trained = false;
const imgSize = 64;

function setup() {
    createCanvas(900, 520);
    initVideo();
    initFaceDetector();
}



function draw() {
    background(100);
    video.loadPixels();
    image(video, 0, 0, 640, 520);
    drawBoxes();
}


function drawBoxes() {
    for (let i = 0; i < boxes.length; i++) {
        const box = boxes[i];
        noFill();
        stroke(161, 95, 251);
        strokeWeight(4);
        rect(box.x, box.y, box.width, box.height);

        if (box.label) {
            fill(161, 95, 251);
            rect(box.x, box.y + box.height, 100, 25);

            fill(255);
            noStroke();
            textSize(18);
            text(box.label, box.x + 10, box.y + box.height + 20);
        }
    }

    // Output
    if (currentFaceImg != null) {
        image(currentFaceImg, 650, 0); // Adjust position as needed
    }

    if (currentEdgeImg != null) {
        image(currentEdgeImg, 650, 300); // Adjust position as needed
    }
}

function initVideo() {
    video = createCapture(VIDEO);
    video.size(640, 520);
    video.hide();
}

function initFaceDetector() {
    faceapi = ml5.faceApi(video, { withLandmarks: true, withDescriptors: false }, () => {
        console.log('Face API Model Loaded!');
        detectFace();
    });
}

function detectFace() {
    faceapi.detect((err, results) => {
        if (err) {
            console.error(err);
            return;
        }

        if (results && results.length > 0) {
            boxes = getBoxes(results);
        }

        detectFace(); // Continue face detection
    });
}

function getBoxes(detections) {
    const newBoxes = [];
    for (let i = 0; i < detections.length; i++) {
        const alignedRect = detections[i].alignedRect;
        const box = {
            x: alignedRect._box._x,
            y: alignedRect._box._y,
            width: alignedRect._box._width,
            height: alignedRect._box._height,
            label: "ID" + boxId
        };

        const similarBox = findSimilarBox(box, prevBoxes);
        if (similarBox) {
            box.label = similarBox.label; // Use the existing ID
        } else {
            boxId += 1;
            box.label = "ID" + boxId; // Assign a new ID
            currentFaceImg = getCroppedImage(box);
            currentEdgeImg = applyEdgeDetection(currentFaceImg);
        }

        newBoxes.push(box);
    }

    prevBoxes = newBoxes;
    return newBoxes;
}

function findSimilarBox(box, prevBoxes) {
    for (let i = 0; i < prevBoxes.length; i++) {
        if (isSimilar(box, prevBoxes[i])) {
            return prevBoxes[i];
        }
    }
    return null;
}

function isSimilar(box1, box2) {
    const center1 = { x: box1.x + box1.width / 2, y: box1.y + box1.height / 2 };
    const center2 = { x: box2.x + box2.width / 2, y: box2.y + box2.height / 2 };

    const distance = dist(center1.x, center1.y, center2.x, center2.y);
    const similarityThreshold = 50;

    return distance < similarityThreshold;
}

function getCroppedImage(box) {
    let img = video.get(box.x, box.y, box.width, box.height);
    return img;
}

function applyEdgeDetection(img) {
    // Convert p5.js image to OpenCV Mat
    let src = cv.imread(img.canvas);

    // Convert the image to grayscale
    let gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    // Apply Canny edge detection
    let edges = new cv.Mat();
    cv.Canny(gray, edges, 50, 150, 3, false);

    // Convert OpenCV Mat back to p5.js image
    let edgeImg = createImage(edges.cols, edges.rows);
    cv.imshow(edgeImg.canvas, edges);
    
    // Release Mats to free memory
    src.delete();
    gray.delete();
    edges.delete();

    return edgeImg;
}

