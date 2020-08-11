# Computer Vision Project
 Yolov3 Evaluation on KITTI dataset
 
## Abstract
Object detection is one of the most vital components in the Automotive sector. It can be used for various applications like Automotive safety, Autonomous driving and so on. Hence, various object detecting models have been developed over the years. One such model is the YOLOv3 which has been developed using the concepts of neural network and deep learning. But once this model is developed, it cannot be directly used in real life applications. The model needs to be evaluated by feeding it datasets containing images with known objects and check the accuracy of detection.  In this project we are trying to evaluate the convolutional neural network YOLOv3 object detector pre trained on COCO dataset. We are trying calculate the precision with which the pre trained YOLOv3 model can detect objects when a completely new dataset is provided to it.  Therefore, this evaluation is done using KITTI dataset for three classes of objects namely Cars, Pedestrians and Trucks. The evaluation is carried out on PyCharm using Python and OpenCV library

## Introduction

### KITTI Dataset
“The KITTI dataset is a recording taken from a moving platform (Car fitted with 4 video cameras, a rotating 3D laser scanner and a combined GPS/IMU inertial navigation system) while driving in and around Karlsruhe, Germany. It includes camera images, laser scans, high-precision GPS measurements and IMU accelerations from a combined GPS/IMU system.” (Andreas Geiger, 2013). It includes camera images of various classes like Cars, Pedestrians, Truck etc. which are 8 in total and also the label files containing ground truth values.

### YOLOv3
YOLO stands for You Only Live Once, it is a convolutional neural network which is used to detect objects in an image. YOLO network has 24 convolutional layers and 2 fully connected layers. It is much faster than a Fast R-CNN (Joseph Redmon, 2016). YOLOv3 is the third improvised version of YOLO neural network which is pre-trained on COCO Dataset and has more convolutional layers for faster detection (Farhadi, 2018).

## KITTI DATASET Contents
From KITTI dataset, 1500 training images and the corresponding label files have been used.
•	KITTI dataset contains 8 different classes which are Car, Van, Truck, Pedestrian, Person sitting, Cyclist, Tram, Misc. and DontCare.
•	Each of these images contains its corresponding label files (.txt format) which are nothing but ground truth values.
•	Each label file consists information of different objects that are present in the image. Class, Truncation, Occlusion, Observation angle, 2D bounding box boundary, 3D bounding box dimension, 3D bounding box coordinates and the Rotation value of the box are information given of each object.

## Evaluation Method
We have to first get the predicted boxes as the output after feeding the images to the YOLOv3 object detector, once we get the predicted boxes, we have to compare these with the ground truth values available in KITTI dataset. The evaluation is done using Mean Average Precision which takes into consideration both Precision and Recall

Average precision can be calculated using the following method.

•	Intersection over Union (IoU).

Intersection over union is based on Jaccard Index. Here, we require both predicted bounding box and ground truth box, intersection over union of the area of both these boxes are calculated. This ratio is nothing but Intersection over Union or IoU. Once we calculate the IoU we proceed to understand whether the detection is valid or not by calculating/assigning True Positives or False Positives with respect to the IoU. If IoU is greater than a certain set threshold then the detection is True Positive else it is False Positive. In the first figure given below there is a small amount of intersection between the Red Box (Ground truth) and Blue Box (Predicted Box) hence the IoU is greater than 0. In the send figure the IoU is equal to 0 as there is no area of intersection. In the third figure there the ground truth box is completely intersected in the predicted  box but as we take the ratio, IoU is greater than 0 but less than 1. Hence, even if there is an intersection as given in third figure, the IoU still needs to be greater than the threshold and such intersection may still not be considered true positive.    
 
ADD IMAGE

Once the no. of True positive’s and False positives are found out, Precision and Recall have to be calculated.

Precision - TP/(TP+FP) (RafaelPadilla, 2020)

Recall-  TP/(TP+FN)    (RafaelPadilla, 2020)

After calculating the above values, we need to plot the cure for Precision and Recall and the area under this curve is Average Precision. 

To find out the area under the curve used the following method has been used.

•	All Point Interpolation:
	AP=∑_(r=0)^N(r_(n+1)-r_n )*ρ(r_(n+1)) (RafaelPadilla, 2020)
	
## Finding the Average Precision using the IoU method per class

### Psudeo Code

GTB- Ground truth boxes and PB- Predicted Boxes
   
		*Initialize a 2-D array of 7000X9 for calculating precision and recall to zero*
   
        *FOR the length of no. of Images (1500 in my case)*
		
               *IF GTB for an image is not 0*
			   
                      *For the length of no. of GTB*
					  
                              *For the length of no. of PB*
							  
                                       *Calculate area of Intersection*
									   
                                       *Calculate area of the union of two boxes* 
									   
                                       *Calculate Intersection over union (IoU)*
									   
                                      *Initialize the IoU to another 2D array*
									  
                   *For the length of the PB (column)*
				   
                                 *Find the maximum ratio across the column and also the row value at which it is maximum.*
                                 
                                      *IF ratio >0.5 (IoU)*
									  
                                           *Find the maximum across the row value*
										   
                                          *IF the maximum in the row value==maximum in column*
										  
                                              *TP=1*
											  
                                               *FP=0*
											   
                                             *Initialize relevant information like Image Name, Class Name,*
											 
                                            *Confidence Score, TP and FP to the 2D array* 
											
                                         *Else*    
										 
                                              *TP=0*
											  
                                              *FP=1*
											  
                                             *Initialize relevant information like Image Name, Class Name,*
											 
                                            *Confidence Score TP and FP to the 2D array*
											
                                    *IF the ratio<0.5 (IoU)*
									
                                          *TP=0*
										  
                                          *FP=1*
										  
                                          *Initialize relevant information like Image Name, Class Name,*
										  
                                         *Confidence Score TP and FP to the 2D array*
										 
           *Sort the 2D array in descending w.r.t confidence score.*
		   
        *FOR the length of no. of rows of the sorted 2D array*
		
                          *Calculate Accumulated TP*
						  
                          *Calculate Accumulated FP*
						  
                          *Calculate Precision with the formula (Accumulated True Positives)/(Accumulated TP+Accumulated FP)*
						  
                         *Initialize the Accumulated TP, Accumulated FP and Precision to the 2D array* 
						 
           *FOR the length of no. of rows of the sorted 2D array*
		   
                          *Calculate Recall with the formula  (Accumulated True Positives)/(Final Accumulated TP+Final Accumulated FP)* 
						  
                         *Initialize the Accumulated TP, Accumulated FP and Precision to the 2D array*
						 
		*Initialize all precision and recall values from the 2D array to two different 1-Darray. With these Precision and Recall values we calculate Average Precision using All point interpolation method.* 

Repeat the same logic for all the selected classes


## Results and Discussion 

Using the above algorithm, the code was run for 1500 images, with IoU threshold being 0.5, no. of True Positives and False Positives, Precision v/s Recall curve and the Average precision was calculated for all the selected classes namely Cars, Pedestrians and Trucks. Using Average Precision per class we calculated Mean Average Precision(mAP) to be 32.59%   

Class Name	 | Average Precision    | No. of True Positives      | No. of False Poistives     | No. of Ground Truth boxes    | No. Predicted Boxes |
:---: 		 | :---         	    | :---                       | :---            			  | :--- 	                     | :---				   |
Car   		 | 60.21       	    	| 4035                		 | 3056      				  | 5583                   		 |7091				   |
Pedestrian   | 27.68				| 338                		 | 443     					  | 893                      	 |781				   |
Truck   	 | 9.87                	| 68	                	 | 608       				  | 215							 |676				   |

	
	

