import numpy as np
import cv2
import os
import sys

# set your data_set absolute path
kitti_img_path = " "##KITTI Image Path
kitti_label_path = " "##KITTI Ground truth Files Path

# transformed lables to save path
kitti_label_tosave_path = " " 


index = 0
cvfont = cv2.FONT_HERSHEY_SIMPLEX

kitti_names = open('kitti.names', 'r')
kitti_names_contents = kitti_names.readlines()
kitti_images = os.listdir(kitti_img_path)
kitti_labels = os.listdir(kitti_label_path)

kitti_images.sort()
kitti_labels.sort()

kitti_names_dic_key = []
for class_name in kitti_names_contents:
    kitti_names_dic_key.append(class_name.rstrip())
values = range(len(kitti_names_dic_key))
kitti_names_num = dict(zip(kitti_names_dic_key, values))

#print(kitti_names_contents)


f = open('train.txt', 'w')
for img in kitti_images:
    f.write(kitti_img_path + img + '\n')
f.close()


for indexi in range(len(kitti_images)):
    kitti_img_totest_path = kitti_img_path + kitti_images[indexi]
    kitti_label_totest_path = kitti_label_path + kitti_labels[indexi]
    # print(kitti_label_totest_path,kitti_img_totest_path)

    kitti_img_totest = cv2.imread(kitti_img_totest_path)
    # print(kitti_img_totest,type(kitti_img_totest))
    img_height, img_width = kitti_img_totest.shape[0], kitti_img_totest.shape[1]

    kitti_label_totest = open(kitti_label_totest_path, 'r')

    label_contents = kitti_label_totest.readlines()
    # print(label_contents)
    real_label = open(kitti_label_tosave_path + kitti_labels[indexi], 'w')

    for line in label_contents:
        data = line.split(' ')
        x = y = w = h = 0
        if (len(data) == 15):
            class_str = data[0]
            if (class_str != 'DontCare'):
                # for kitti calls is a string
                # trans this to number by using kitti.names
                # (x,y) center (w,h) size
                x1 = float(data[4])
                y1 = float(data[5])
                x2 = float(data[6])
                y2 = float(data[7])

                intx1 = int(x1)
                inty1 = int(y1)
                intx2 = int(x2)
                inty2 = int(y2)

                bbox_center_x = float((x1 + (x2 - x1) / 2.0) / img_width)
                bbox_center_y = float((y1 + (y2 - y1) / 2.0) / img_height)
                bbox_width = float((x2 - x1) / img_width)
                bbox_height = float((y2 - y1) / img_height)
                line_to_write = str(class_str) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(
                    x2) + ' ' + str(y2) + '\n'
                real_label.write(line_to_write)
                sys.stdout.write(str(int((indexi / len(kitti_images)) * 100)) + '% ' + '*******************->' "\r")
                sys.stdout.flush()
    real_label.close()
kitti_names.close()
print("Labels tranfrom finished!")