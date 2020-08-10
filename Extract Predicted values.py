import cv2
import numpy as np
import os
import glob
import random
import matplotlib.pyplot as plt

# Load Yolo
net = cv2.dnn.readNet("D:\darknet-master\cfg/yolov3.weights", "D:\darknet-master\cfg/yolov3.cfg") ##.weights of the neural netwotk and configuration file path
#classes = []

classes = ["Pedestrian","bicycle","Car","motorbike","aeroplane","bus","train","Truck","boat","traffic_light","fire_hydrant","stop_sign","parking_meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports_ball","kite","baseball_bat","baseball_glove","skateboard","surfboard","tennis_racket","bottle","wine_glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa","potted_plant","bed","dining_table","toilet","tv_monitor","laptop","mouse","remote","keyboard","cell_phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy_bear","hair_drier","toothbrush"]
#with open("coco.names","r") as f:
#    classes = [line.strip() for line in f.readlines()]
# Name custom object
#classes = ["person","bicycle","car","motorbike"]

# Images path
# images_path = "D:\Kitti\extraction\training\image_2/000000.jpg"

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#file1=''
def load_images(path="D:\\CV experiment\images 1000-1500/"):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]

file_names=load_images()
#print(file_names)
images = []
image_name=[]
c=''
for file in file_names:
     c=os.path.basename(file)
     image_name.append((os.path.splitext(c)[0][0:]))
     images.append(cv2.imread(file,cv2.IMREAD_UNCHANGED))

print(image_name)

# Detecting objects
#for i in range(len(images)):
# blob[i] = cv2.dnn.blobFromImage(images[i], 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#
# print(blob)
# net.setInput(blob)
# outs = net.forward(output_layers)
# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
blob=[0]*len(images)
x=0.0
y=0.0
w=0.0
h=0.0
extension='txt'
text_name=''
for i in range(len(images)):
    blob[i] = cv2.dnn.blobFromImage(images[i], 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    #print(blob)
    net.setInput(blob[i])
    outs = net.forward(output_layers)
    height = images[i].shape[0]
    width = images[i].shape[1]
    channels = images[i].shape[2]
    text_name=image_name[i]+'.txt'
    fh = open(text_name, 'w')
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            #print(scores)
            #fh = open(text_name, 'w')
            if confidence > 0.3:
            # Object detected
               #print(class_id)
               center_x = int(detection[0] * width)
               center_y = int(detection[1] * height)
               w = int(detection[2] * width)
               h = int(detection[3] * height)
            # Rectangle coordinates
               x = int(center_x - w / 2)
               y = int(center_y - h / 2)
               #open_file()
               fh = open(text_name, 'a')
               label = str(classes[class_id])
               fh.write("%s " % (label))
               fh.write("%s " % (float(confidence)))#(class_id))
               fh.write("%i " % (x))
               fh.write("%i " % (y))
               fh.write("%i " % (x + w))
               fh.write("%i" % (y + h))
               fh.write("\n")
               fh.close()
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        #print(indexes)
        #font = cv2.FONT_HERSHEY_PLAIN
        # for g in range(len(boxes)):
        #     if g in indexes:
        #        x, y, w, h = boxes[g]
        #        #print(x, y, x + w, y + h, confidences[g])
        #        #for out in outs:
        #     fh= open(text_name, 'w')
        #     fh.write("%i " % (class_id))
        #     fh.write("%i " % (x))
        #     fh.write("%i " % (y))
        #     fh.write("%i " % (x + w))
        #     fh.write("%i " % (y + h))
        #     fh.write("%s " % (str(confidences[g])))
        #     fh.close()


#        label = str(classes[class_ids[i]])
#        color = colors[class_ids[i]]
#        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#        cv2.putText(img, label, (x, y + 30), font, 0.5, color, 2)
#        print(x, y, x + w, y + h, confidences[i])
#    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
#    cv2.imshow("Image", img)
#    cv2.waitKey(10000)
#    cv2.destroyAllWindows()