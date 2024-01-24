let video, classifier, faceapi, inputLabel;
let boxes = [];
let boxId = 0;
let prevBoxes = [];

let trained = false;
const imgSize = 64;

function setup() {
    createCanvas(900, 520);

    initVideo();
    initFaceDetector();
}

function draw() {
    background(0);
    image(video, 0, 0, 640, 520)
    drawBoxes()
}

function drawBoxes() {
    for (let i = 0; i < boxes.length; i++) {
        const box = boxes[i]
        noFill()
        stroke(161, 95, 251)
        strokeWeight(4)
        rect(box.x, box.y, box.width, box.height)

        if (box.label) {
            fill(161, 95, 251)
            rect(box.x, box.y + box.height, 100, 25)

            fill(255)
            noStroke()
            // strokeWeight(2)
            textSize(18)
            text(box.label, box.x + 10, box.y + box.height + 20)
        }
    }
}

function initVideo() {
    video = createCapture(VIDEO)
    video.size(640, 520)
    video.hide()
}

function initFaceDetector() {
    const detectionOptions = {
        withLandmarks: true,
        withDescriptors: false
    };

    faceapi = ml5.faceApi(video, detectionOptions, () => {
        console.log('Face API Model Loaded!')
        detectFace()
    });
}

function detectFace() {
    faceapi.detect((err, results) => {
        if (err) return console.error(err)

        boxes = []
        if (results && results.length > 0) {
            boxes = getBoxes(results)
        }
        detectFace()
    })
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

        // Check for a similar box in prevBoxes
        const similarBox = findSimilarBox(box, prevBoxes);
        if (similarBox) {
            box.label = similarBox.label; // use the existing ID
        } else {
            boxId += 1;
            box.label = "ID" + boxId; // assign a new ID
        }

        newBoxes.push(box);
    }

    prevBoxes = newBoxes; // update prevBoxes for the next frame
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
    // Calculate the center point of each box
    const center1 = { x: box1.x + box1.width / 2, y: box1.y + box1.height / 2 };
    const center2 = { x: box2.x + box2.width / 2, y: box2.y + box2.height / 2 };

    // Calculate Euclidean distance between the two centers
    const distance = dist(center1.x, center1.y, center2.x, center2.y);

    // Define a threshold for considering boxes as similar
    const similarityThreshold = 50; // This value can be adjusted

    // Return true if the distance is less than the threshold
    return distance < similarityThreshold;
}
