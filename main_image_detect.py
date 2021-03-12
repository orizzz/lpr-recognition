import cv2
import numpy as np
from function import crop_objects, recognize, write_image

#load YOLO network
weight = "./yolov3/yolov4-spn.weights"
config = "./yolov3/yolov4-spn.cfg"
classes_name = "./yolov3/plate.names"
data_image = "data/motor1.jpg"
save_path = "data/detected/"
input_size = 416

def run():
    net = cv2.dnn.readNet(weight,config)
    classes = []
    with open(classes_name, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_Layer = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    np.random.seed(12)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    #loading image
    img = cv2.imread(data_image)
    img = cv2.resize(img,None,fx=1,fy=1)
    # img = cv2.resize(img, (input_size, input_size))

    height, width, channels = img.shape

    #detect object
    blob = cv2.dnn.blobFromImage(img,0.0039,(416,416),(0,0,0),True,crop=False)
    net.setInput(blob)
    outs = net.forward(output_Layer)

    # Showing information on screen
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # object detected

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinate
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                ymin = y
                xmin = x
                ymax = y + h
                xmax = x + w

                boxes.append([xmin, ymin, xmax, ymax])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            xmin, ymin, xmax, ymax = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            crop_image = crop_objects(img, xmin, ymin, xmax, ymax)
            plate_num = recognize(crop_image)
            write_image(crop_image, save_path, class_name=label, cropped=True)

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(img, label, (x+5, y+15), font, 1, color, 2)
            write_image(img, save_path, class_name=label)


    # cv2.imshow("image",img)
    # cv2.waitKey(0)
    # cv2.destroyAllwindwos()

run()