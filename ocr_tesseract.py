import os
import cv2
import pytesseract
from ultralytics import YOLO
import csv

# Function to perform license plate detection and recognition
def detect_and_recognize_license_plate(frame, car_id, model, csv_file):
    results = model(frame) 

    if results:
        for result in results[0]:
            if result[-1] == 0:
                x1, y1, x2, y2 = map(int, result[:4])
                license_plate_region = frame[y1:y2, x1:x2]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Perform OCR on the license plate region to recognize text
                text = pytesseract.image_to_string(license_plate_region, config='--psm 11').strip()

                # Draw text on the frame with the recognized license plate number
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print("Text detected:", text)
                print("Car ID:", car_id)
    
                # Check if text is recognized
                if text:
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([text, car_id])  # Swap the order of columns
                    
                    # Draw bounding box around the detected license plate
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # # Draw text on the frame with the recognized license plate number
                    # cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # print("Text detected:", text)
                    # print("Car ID:", car_id)
    
    return frame

# Function to read video and process frames
def process_video(input_video, model):
    cap = cv2.VideoCapture(input_video)
    car_id = 0
    csv_file = "car_ids.csv"
    
    # Get the full path of the CSV file
    csv_file_path = os.path.abspath(csv_file)

    # Clear CSV file if it already exists
    if os.path.exists(csv_file):
        os.remove(csv_file)
    print("CSV file created at:", csv_file)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        car_id += 1

        frame = detect_and_recognize_license_plate(frame, car_id, model, csv_file)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("CSV file path:", csv_file_path)
    return csv_file_path  # Return the full path of the CSV file

if __name__ == "__main__":
    # Load your custom object detection model (replace this with your model loading code)
    coco_model = YOLO("yolov8n.pt")
    model = YOLO("best.pt")

    # Specify the input video file
    input_video = "video.mp4"

    # Process the video frames
    csv_file_path = process_video(input_video, model)
    print("CSV file saved at:", csv_file_path)
