import cv2
import numpy as np
import time
from scipy.spatial import distance

confThreshold = 0.5
nmsThreshold = 0.2
inpWidth = 416
inpHeight = 416

cfg_path = '../yolo_files/yolov3-tiny.cfg'
weights_path = '../yolo_files/yolov3-tiny_6000.weights'

net = cv2.dnn.readNetFromDarknet(cfg_path , weights_path)

classes = []
with open("../yolo_files/obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def discrete_color(r,g,b):
    distances = []
    colors = [[255, 255, 255],
              [160, 160, 160], 
              [64, 64, 64], 
              [127, 000, 255], 
              [000, 000, 255], 
              [51, 153, 255], 
              [51, 255, 51], 
              [255, 255, 000], 
              [255, 153, 51], 
              [255, 000, 000]]
    for color in colors:
        distances.append(distance.euclidean(color, (r,g,b)))
    return colors[distances.index(min(distances))]

def detect_color(image):
    image = image[image.shape[0]//10:-image.shape[0]//3, image.shape[1]//5:-image.shape[1]//5]
    pixel_vals = np.float32(image.reshape((-1,3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    try:
        _, _, centers = cv2.kmeans(pixel_vals, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = centers[0]
    except:
        centers = [0, 0, 0]
    return centers

def drawPred(classId, conf, left, top, right, bottom):
# Draw the predicted bounding box
    obj_class = classes[classId]
    label = '%.2f' % (conf*100)
    label = '%s:%s' % (classes[classId], label)
    r, g, b = 0, 0, 0

    if obj_class == 'helmet':
        [b, g, r] = detect_color(img[top:bottom, left:right])
        #[r, g, b] = discrete_color(r, g, b)
    
    cv2.rectangle(img, (left, top), (right, bottom), (int(b), int(g), int(r)), 2)    

    #Display the label at the top of the bounding box
    cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

def postprocess(img, outs):
    frameHeight, frameWidth = img.shape[0], img.shape[1]

    classIds, confidences, boxes = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]

        drawPred(classIds[i], confidences[i], box[0], box[1], box[0] + box[2], box[1] + box[3])

fourcc = 0x00000021
writer = cv2.VideoWriter('test.mp4', fourcc, 20, (1280, 720))
cap = cv2.VideoCapture('../media/vid.mp4')

ret = True
while ret:
    start = time.time()
    ret, frame = cap.read()

    img = cv2.resize(frame, None, fx=1.0, fy=1.0)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    postprocess(img, outs)

    end = time.time()
    writer.write(img)
    fps = 1/(end - start)
    print(fps)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()