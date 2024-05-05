import re
import cv2
import csv
import string
import easyocr
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt


reader = easyocr.Reader(['en'], gpu=True)


coco_model = YOLO("yolov8n.pt")
model = YOLO(r"D:\\Automatic_Number_Plate_Recognition\\best.pt")


int_to_char = {'0': 'O',
               '1': 'I',
               '2': 'Z',
               '3': 'B',
               '4': 'A',
               '5': 'S',
               '6': 'b',
               '7': 'T',
               '8': 'B',
               '9': 'q'}

char_to_int = {'A': '4',
               'B': '8',
               'b': '6',
               'D': '0',
               'G': '6',
               'g': '9',
               'I': '1',
               'J': '7',
               'L': '4',
               'l': '1',
               'O': '0',
               'o': '0',
               'q': '9',
               'S': '5',
               's': '5',
               'T': '7',
               'Z': '2',
               'z': '2'}
    

# Function for inference in real time
def real_time():
    
    lp = ""
    frm_nm = -1
    cap = cv2.VideoCapture("D:\\Automatic_Number_Plate_Detection\\video.mp4")
    ret = True

    with open("results1.csv", mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Frame Number', 'Timestamp', 'License Plate Text'])

        while ret:
            ret, frame = cap.read()  # Read frame
            frm_nm += 1

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = model(frame)[0]  # Detect license plates

                for detection in detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection  # License plate coordinates
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    score = np.round(score, 2)

                    if score >= 0.42:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        crop = frame[y1:y2, x1:x2]  # Cropped license plate image

                        dat = perform_ocr(crop)
                        for bbox, text, confidence in dat:
                            '''
                            Perform a check on text to see if len(text) is > 7
                            Because Indian License Plates format = LL DD LL DDDD nowadays
                            Ignore all special characters and blank spaces
                            '''
                            text = re.sub(r'[^a-zA-Z0-9]', '', text)
                            print(text)
                            if len(text) >=4 and len(text) <=6: # To check for 2 line license plates
                                p1 = text
                                lp += p1

                            elif len(text) == 9:
                                lp = text
                            
                            elif len(text) == 10:
                                lp = text

                            text = lp.upper()
                            print("Plate = ",text, "\nCount = ",len(text))
                            '''
                            Add new check for numbers plates with 8 and 9 characters as well
                            '''                            
                            if len(text) == 10:
                                if license_complies_format(text):
                                    print(format_license(text))
                                    timestamp = datetime.now()
                                    cv2.putText(frame, f'{format_license(text)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                                    write_csv(csv_writer, frm_nm, timestamp, format_license(text))
                            
                            elif len(text) == 9:
                                if license_complies_format(lp):
                                    print(format_license(lp))
                                    timestamp = datetime.now()
                                    cv2.putText(frame, f'{format_license(lp)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                                    write_csv(csv_writer, frm_nm, timestamp, format_license(lp))

                            else:
                                continue

                            # if len(text) != 10:
                            #     continue
                            # else:
                            #     if license_complies_format(text):
                            #         print(format_license(text))
                            #         timestamp = datetime.now()
                            #         write_csv(csv_writer, frm_nm, timestamp, format_license(text))

                cv2.imshow("Frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                # Press 'q' to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            else:
                print("End of Frames!")

        # Release the capture and close all windows
        cap.release()
        cv2.destroyAllWindows()




def perform_ocr(frame):  # OCR function
    ocr = reader.readtext(frame)  # Performing OCR on the license plate
    return ocr


def write_csv(csv_writer, frm_nm, time, text):  # Write OCR text to CSV
    csv_writer.writerow([frm_nm, time, text])


# Function to check license plate format
bool = False
def license_complies_format(text):
    if len(text) == 10:
        # For 10 character license plates
        if (text[0] in string.ascii_uppercase or text[0] in int_to_char.keys()) and \
                (text[1] in string.ascii_uppercase or text[1] in int_to_char.keys()) and \
                (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in char_to_int.keys()) and \
                (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in char_to_int.keys()) and \
                (text[4] in string.ascii_uppercase or text[4] in int_to_char.keys()) and \
                (text[5] in string.ascii_uppercase or text[5] in int_to_char.keys()) and \
                (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in char_to_int.keys()) and \
                (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[7] in char_to_int.keys()) and \
                (text[8] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[8] in char_to_int.keys()) and \
                (text[9] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[9] in char_to_int.keys()):
                bool = True
                return bool
        
    elif len(text) == 9:
        #For 9 character license plates+
        if (text[0] in string.ascii_uppercase or text[0] in int_to_char.keys()) and \
                (text[1] in string.ascii_uppercase or text[1] in int_to_char.keys()) and \
                (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in char_to_int.keys()) and \
                (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in char_to_int.keys()) and \
                (text[4] in string.ascii_uppercase or text[4] in int_to_char.keys()) and \
                (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in char_to_int.keys()) and \
                (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in char_to_int.keys()) and \
                (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[7] in char_to_int.keys()) and \
                (text[8] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[8] in char_to_int.keys()):
                bool = True
                return bool


# Function to reformat license plate into expected correct format
def format_license(text):
    license_plate_ = ''

    if len(text) == 10:
        # For 10 character license plates
        mapping = {0: int_to_char, 1: int_to_char, 4: int_to_char, 5: int_to_char,
                2: char_to_int, 3: char_to_int, 6: char_to_int, 7: char_to_int, 8: char_to_int, 9: char_to_int}
        for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]
        return license_plate_
    
    elif len(text) == 9:
        mapping = {0: int_to_char, 1: int_to_char, 4: int_to_char,
                2: char_to_int, 3: char_to_int, 5: char_to_int, 6: char_to_int, 7: char_to_int, 8: char_to_int}
        for j in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]
        return license_plate_
    

real_time()