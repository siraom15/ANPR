import cv2
import datetime
import easyocr
from imutils.video import VideoStream
import time

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

def recognize_plate(image):
    results = reader.readtext(image)
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # If confidence is high enough, consider it a license plate
        if prob > 0.5: 
            return text
    return None

def main():
    # Start the video stream
    vs = cv2.VideoCapture(0)
    # vs = cv2.VideoCapture('')
    time.sleep(2.0)  # Allow the camera sensor to warm up

    while True:
        # Capture frame-by-frame
        frame = vs.read()

        # Recognize plate in the frame
        plate = recognize_plate(frame)

        if plate:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f'License Plate: {plate} | Date-Time: {current_time}')

        # Display the frame
        cv2.imshow('Frame', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Add a delay of 500ms between frame analyses
        # time.sleep(0.5)

    # When everything done, release the capture
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
