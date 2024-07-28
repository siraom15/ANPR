import cv2
import time
import os
import easyocr
import datetime

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

def init_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def recognize_plate(image):
    results = reader.readtext(image)
    for (bbox, text, prob) in results:
        if prob > 0.5: 
            return text
    return None

def main():
    output_dir = 'car-image'
    init_dir(output_dir)

    # Initiate video capture from the specified video file
    cap = cv2.VideoCapture('D:/lpr/poc/data/2.mp4')

    # Allow the camera sensor to warm up
    time.sleep(2.0)

    # Loop once video is successfully loaded
    while cap.isOpened():
        time.sleep(0.05)
        ret, frame = cap.read()

        if not ret:
            break

        plate = recognize_plate(frame)

        if plate:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f'License Plate: {plate} | Date-Time: {current_time}')

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) == 13:  # Break loop on 'Enter' key press
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
