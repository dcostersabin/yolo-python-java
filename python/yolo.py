import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path

current = os.path.dirname(os.path.abspath(__file__))
parent = Path(current).parent
yolo_weights = f'{parent}/Assets/models/yolov3.weights'
yolo_cfg = f'{parent}/Assets/models/yolov3.cfg'
coco_names = f'{parent}/Assets/models/coco.names'


def load_yolo():
    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    classes = []

    with open(coco_names, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layers_names = net.getLayerNames()
    unconnected_layer = net.getUnconnectedOutLayers()
    output_layers = [layers_names[i - 1] for i in unconnected_layer]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers


def load_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, None, fx=0.9, fy=0.9)
    height, width, channels = image.shape
    return image, height, width, channels


def detect_object(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(608, 608), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []

    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.5:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)

    return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    cv2.imwrite(f'{parent}/outputimages/python/{datetime.now()}.jpg', img)


def image_detect(img_path):
    model, classes, colors, output_layers = load_yolo()
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_object(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    draw_labels(boxes, confs, colors, class_ids, classes, image)



if __name__ == "__main__":
    files = []
    for(path,name,filenames) in os.walk(f'{parent}/testimages/'):
        for filename in filenames:
            if filename.endswith('.jpg'):
                image_detect(f'{path}{filename}')

