import cv2
import numpy as np
import time
import os

def init_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def main():
    output_dir = 'lcp-images'
    init_dir(output_dir)

    frame_count = 0

    # Load the pre-trained Haar Cascade classifier for Russian car plates
    carPlatesCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

    # Open the video file
    cap = cv2.VideoCapture('data/5.mp4')

    # Set frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 80)

    # Wait for the camera to warm up
    time.sleep(2.0)

    # Check if the video capture opened successfully
    if not cap.isOpened():
        print('Error reading video')
        return

    print('Video opened successfully:', cap.isOpened())

    while cap.isOpened():
        time.sleep(0.05)
        # Read frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect car plates in the frame
        plate = carPlatesCascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=2)

        if len(plate) > 0:
            # Draw rectangles around detected car plates
            for (x, y, w, h) in plate:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            # Save the frame with rectangles to the specified folder
            frame_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            
        # Display the resulting frame
        cv2.imshow('Cars', frame)

        # Break loop on 'Enter' key press
        if cv2.waitKey(1) == 13:
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
