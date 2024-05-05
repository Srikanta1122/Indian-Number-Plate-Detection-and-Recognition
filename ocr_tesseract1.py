import os
import cv2
import pytesseract
from ultralytics import YOLO
import csv

# Function to perform license plate detection and recognition
def detect_and_recognize_license_plate(frame, car_id, csv_writer):
    # Your YOLO detection code here
    coco_model = YOLO("yolov8n.pt")
    model = YOLO("best.pt")
    # For demonstration, let's assume text recognition output
    text = pytesseract.image_to_string(frame, config='--psm 11').strip()
    print("Text:", text)

    # Check if text is recognized
    if text:
        # Write to CSV
        csv_writer.writerow([text, car_id])
        print("Text detected:", text)
        print("Car ID:", car_id)

    return frame

# Function to read video and process frames
def process_video(input_video):
    cap = cv2.VideoCapture(input_video)
    car_id = 0
    csv_file = "car_ids.csv"

    # Open CSV file in write mode
    with open(csv_file, mode='w', newline='') as file:
        csv_writer = csv.writer(file)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            car_id += 1

            frame = detect_and_recognize_license_plate(frame, car_id, csv_writer)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    csv_file_path = os.path.abspath(csv_file)
    print("CSV file path:", csv_file_path)
    return csv_file_path

if __name__ == "__main__":
    # Specify the input video file
    input_video = "video.mp4"

    # Process the video frames
    csv_file_path = process_video(input_video)
    print("CSV file saved at:", csv_file_path)
