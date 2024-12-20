import cv2
import numpy as np
from fer import FER
import logging
from datetime import datetime

class SmartDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.emotion_detector = FER(mtcnn=True)
        self.emotion_threshold = 0.5
        
        logging.basicConfig(
            filename=f'smart_detector_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.cap = None
    
    def start_video_capture(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logging.error("Failed to open video capture")
            raise RuntimeError("Could not start video capture")
        logging.info("Video capture started successfully")
    
    def detect_faces(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces
        except Exception as e:
            logging.error(f"Error in face detection: {str(e)}")
            return []
    
    def detect_emotion(self, frame):
        try:
            emotions = self.emotion_detector.detect_emotions(frame)
            if emotions:
                dominant_emotion = max(
                    emotions[0]['emotions'].items(),
                    key=lambda x: x[1]
                )
                if dominant_emotion[1] >= self.emotion_threshold:
                    return {
                        'emotion': dominant_emotion[0],
                        'probability': dominant_emotion[1],
                        'box': emotions[0]['box']
                    }
            return None
        except Exception as e:
            logging.error(f"Error in emotion detection: {str(e)}")
            return None
    
    def process_frame(self, frame):
        faces = self.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        emotion_data = self.detect_emotion(frame)
        
        if emotion_data:
            box = emotion_data['box']
            emotion = emotion_data['emotion']
            probability = emotion_data['probability']
            
            cv2.rectangle(
                frame,
                (box[0], box[1]),
                (box[0] + box[2], box[1] + box[3]),
                (0, 255, 0),
                2
            )
            
            cv2.putText(
                frame,
                f"{emotion}: {probability:.2f}",
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
        
        return frame, faces, emotion_data
    
    def run(self):
        self.start_video_capture()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("Failed to grab frame")
                    break
                
                frame, faces, emotion_data = self.process_frame(frame)
                
                cv2.imshow('Smart Detection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            logging.error(f"Runtime error: {str(e)}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        logging.info("Detection system shutdown complete")

if __name__ == "__main__":
    detector = SmartDetector()
    try:
        detector.run()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        detector.cleanup()