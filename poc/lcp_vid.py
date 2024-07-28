import cv2
import numpy as np
import time
import os

def init_dir(output_dir):
    print("Initing Dir: "+ output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Successful Init: "+ output_dir)
    else:
        print(output_dir + " is exist.")

def main():
    # Init dir
    output_dir = 'lcp-images'
    init_dir(output_dir)


    frame_count = 0
    frame_skip = 5

    # Load the pre-trained Haar Cascade classifier for Russian car plates
    carPlatesCascade = cv2.CascadeClassifier('haarcascade/haarcascade_russian_plate_number.xml')

    # Open the video file
    cap = cv2.VideoCapture('test-data/5.mp4')

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

        # Skip frames to reduce the rate of capture
        if frame_count % frame_skip == 0:
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Detect car plates in the frame
            plates = carPlatesCascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=2, minSize=(30, 30))

            if len(plates) > 0:
                # Draw rectangles around detected car plates
                for (x, y, w, h) in plates:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                # Save the frame with rectangles to the specified folder
                frame_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
                cv2.imwrite(frame_path, frame)

        # Display the resulting frame
        cv2.imshow('Cars', frame)
        frame_count += 1

        # Break loop on 'Enter' key press
        if cv2.waitKey(1) == 13:
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
