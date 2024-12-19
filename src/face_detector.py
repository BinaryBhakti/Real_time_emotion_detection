import cv2
import numpy as np
import logging
from datetime import datetime

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        logging.basicConfig(
            filename = f'smart_mirror_{datetime.now().strftime("%Y%m%d")}.log',
            level = logging.INFO,
            format = '%(asctime)s - %(levelname)s - %(message)s'
        )

        self.cap = None

    
    def start_video_capture(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logging.error("Failed to open video capture")
            raise RuntimeError("Could not start video capture")


    def detect_faces(self, frame):
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor = 1.1,
                minNeighbors = 5,
                minSize = (30, 30)
            )
            return faces
        except Exception as e:
            logging.error(f"Error in detect_faces: {e}")
            return []


    def process_frame(self, frame):
        faces = self.detect_faces(frame)

        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return frame, faces

    
    def run(self):

        self.start_video_capture()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to grab frame")
                breaak


            frame, faces = self.process_frame(frame)

            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()


    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = FaceDetector()
    try:
        detector.run()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        detector.cleanup()