
# Indian-Number-Plate-Detection-and-Recognition: Step-By-Step guide for use this project

## 🔗 Social Media Links:
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/srikantapramanik/)
## Re-Train YOLO Model for Licence Plate Detection: 

First of all from the roboflow using API key download the dataset. The dataset contains train, test and the valid folder, inside these folder there is a actual images of the licence plate as .jpg format and have there levels as a .txt format. The dataset already present there with the annotation format then using that dataset pre-train the YOLO model. After the training process is completed is return best.pt and last.pt as model which can detect the licence plate.  
#### Get the dataset from roboflow:
[Licence-Plate-Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4) 
#### For train the YOLO model use these code: 
[ReTrain-YOLOv8-for-Number-Plate-Detection](https://github.com/Srikanta1122/Indian-Number-Plate-Detection-and-Recognition/blob/master/TrainYOLOv8_By_Licence_Plate_Dataset.ipynb)

Inside the data.yaml file give your actual train, test and valid images path. After the training process is completed it will return best.pt and last.pt that can detect the licence plate from the images and videos. 


## Libraries Used:

	OpenCV (cv2): For image processing tasks like reading frames from video, image conversion, and drawing bounding boxes. 

	EasyOCR: For performing Optical Character Recognition (OCR) on license plate images.

	NumPy: For numerical operations on arrays.

	Ultralytics YOLO: For object detection, specifically for detecting license plates.

	Matplotlib: For visualizing results and debugging.

## Model Details:

Two YOLO models are used for this project:

i)	License Plate Detection Model (model): This model is loaded from a pre-trained weights file ("best.pt") and is used to detect license plates in the frames.

ii)	General Object Detection Model (coco_model): This model is loaded from a pre-trained weights file ("yolov8n.pt") and is used as a fallback or as an alternative for detecting license plates if the primary model fails.

## Workflow: 
1)	Initialize EasyOCR reader for English text recognition.
2)	Load YOLO models for license plate detection and general object detection.
3)	Define dictionaries for mapping characters to integers and vice versa.
4)	Define functions for real-time processing, OCR, CSV writing, license plate format compliance check, and license plate formatting.
5)	Open a video file for processing.
6)	Initialize variables and set up a CSV file for storing results.
7)	Loop through frames from the video:
	Read each frame.

	Convert the frame to RGB format.

	Use the primary license plate detection model to detect license plates in the frame.

	For each detected license plate:

    o	Extract the coordinates of the bounding box.
    o	Crop the license plate region from the frame.
    o	Perform OCR on the cropped license plate image.
    o	Clean and format the OCR text.
    o	Check if the text matches a valid license plate format.
    o	If valid, record the frame number, timestamp, license plate text, coordinates, and confidence score to the CSV file.
	Display the frame with bounding boxes and recognized text.

	If the 'q' key is pressed, break the loop. 

8)	Once all frames are processed, release the video capture and close all windows.


## Run Locally:

Clone the repository

```bash
  git clone https://github.com/Srikanta1122/Indian-Number-Plate-Detection-and-Recognition.git
```

Install dependencies

```bash
  install all the required libery that I mention on the top. 
```

Go to the final_ocr1.py file 

```bash
  "First, edit the file path. give youe actual file path for model('best.pt' or 'last.pt') and input  
   video. after that run this file". 
```

Result

```bash
  "after read of the input video frame-by-frame it will return a .CSV file which contains the  
  ['Frame_number', 'Timestamp', 'License_plate_text', 'License_plate_coordinates', 'Confidence_score']   
   as a column".  
```

## Final Result: 
Look into the below CSV file which contains Frame_number, Timestamp, License_plate_text, License_plate_coordinates and the Confidence_score. 

[CSV-File](https://github.com/Srikanta1122/Indian-Number-Plate-Detection-and-Recognition/blob/master/results1.csv)

Inside the CSV file, list of License_plate_text is present for the multiple car id. But in this case same number plate Recognize multiple time with the different text format. 

