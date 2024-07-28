import cv2
import time
import os

# Create our car classifier
car_classifier = cv2.CascadeClassifier('cars.xml')

def init_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def main():
    output_dir = 'car-image'
    init_dir(output_dir)

    # Initiate video capture from the facecam (index 0)
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('D:/lpr/poc/data/4.mp4')

    # Allow the camera sensor to warm up
    time.sleep(2.0)

    frame_count = 0  # Initialize frame count for naming saved images

    # Loop once video is successfully loaded
    while cap.isOpened():
        time.sleep(0.05)
        # Read frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pass frame to our car classifier
        cars = car_classifier.detectMultiScale(gray, 1.4, 2)

        # Extract bounding boxes for any cars identified
        if len(cars) > 0:
            # Draw rectangles around detected cars
            for (x, y, w, h) in cars:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Save the frame with rectangles to the specified folder
            frame_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        cv2.imshow('Cars', frame)

        if cv2.waitKey(1) == 13:  # Break loop on 'Enter' key press
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()