import cv2
import numpy as np
import argparse
import os
from imutils.video import FPS

ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("--detect", type=str, default="image",
	help="Detect image or video")
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when apply non-maxima suppression")
ap.add_argument("-u", "--use-gpu", type=bool, default=0,
	help="boolean indicating if CUDA GPU should be used")
args = vars(ap.parse_args())

# Yolo file
yolo_label = "obj.names"
yolo_weight = "yolov3_custom.weights"
yolo_conf = "yolov3_custom.cfg"
labelsPath = os.path.sep.join([args["yolo"], yolo_label])
weightsPath = os.path.sep.join([args["yolo"], yolo_weight])
configPath = os.path.sep.join([args["yolo"], yolo_conf])

# Membaca file label
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))

print("[INFO] loading YOLO configuration...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if args["use_gpu"]:
	print("[INFO] setting backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


if args["detect"] == "video":
	print("[INFO] Detecting Video...")
	vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
	outputVideo = None
	fps = FPS().start()

	while True:
		(grabbed, frame) = vs.read()
		if not grabbed:
			break

		(H, W) = frame.shape[:2]

		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
		net.setInput(blob)
		layerOutputs = net.forward(layer_names)

		boxes = []
		confidences = []
		classIDs = []

		for output in layerOutputs:
			for detection in output:

				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				if confidence > args["confidence"]:

					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

		if len(idxs) > 0:
			for i in idxs.flatten():
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.1f}%".format(LABELS[classIDs[i]],
					confidences[i] * 100)
				cv2.putText(frame, text, (x, y - 5),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		
		if args["display"] > 0:
			# show the output frame
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break

		if args["output"] != "" and outputVideo is None:
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			outputVideo = cv2.VideoWriter(args["output"], fourcc, 30,
				(frame.shape[1], frame.shape[0]), True)
				
		if outputVideo is not None:
			outputVideo.write(frame)
		fps.update()

	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

elif args["detect"] == "image":
	print("[INFO] Detecting Image...")
	img = cv2.imread(args["input"] if args["input"] else 0)
	# img = cv2.resize(img,None,fx=0.3,fy=0.3)

	(H, W) = img.shape[:2]

	blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(layer_names)
	
	boxes = []
	confidences = []
	classIDs = []

	for output in layerOutputs:
		for detection in output:

			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > args["confidence"]:
				
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				print(detection)
				print(detection[0:4])
				print(detection[0:4] * np.array([W, H, W, H]))
				print(detection[5:])
				print(float(confidence)*100)

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.1f}%".format(LABELS[classIDs[i]],confidences[i] * 100)
			cv2.putText(img, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	if args["display"] > 0:
		# show the output image
		cv2.imshow("Image", img)
		key = cv2.waitKey(0) & 0xFF
	
	if args["output"] != "":
		cv2.imwrite(args["output"],img)
			