import cv2
import dlib
from imutils import face_utils
import numpy as np
import imutils
net = cv2.dnn.readNetFromDarknet("yolov3.cfg","yolov3_1100.weights")
layer_names = net.getLayerNames()
print("成功")
# print(layer_names[199])
# print(layer_names[226])
print(layer_names[250])
print("成功")
for i in net.getUnconnectedOutLayers():
    print(i[0])
print("成功")
print(net.getUnconnectedOutLayers())
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
classes = [line.strip() for line in open("obj.names")]
colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # forward propogation
        img = cv2.resize(image, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # get detection boxes
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                tx, ty, tw, th, confidence = detection[0:5]
                scores = detection[5:]
                class_id = np.argmax(scores)
                if confidence > 0.3:
                    center_x = int(tx * width)
                    center_y = int(ty * height)
                    w = int(tw * width)
                    h = int(th * height)

                    # 取得箱子方框座標
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # draw boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                cv2.putText(img, label, (x, y - 5), font, 3, color, 2)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
