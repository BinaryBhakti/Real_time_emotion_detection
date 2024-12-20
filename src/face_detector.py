import cv2
import numpy as np
from fer import FER
import logging
from datetime import datetime
from pathlib import Path
import random
import json

class SmartDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.emotion_detector = FER(mtcnn=True)
        self.emotion_threshold = 0.5
        self.compliments = self._load_compliments()
        self.used_compliments = set()
        
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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        emotion_data = self.detect_emotion(frame)

        if emotion_data:
            box = emotion_data['box']
            emotion = emotion_data['emotion']
            probability = emotion_data['probability']

            if box and len(box) == 4:
                cv2.rectangle(
                    frame,
                    (max(0, box[0]), max(0, box[1])),
                    (min(frame.shape[1], box[0] + box[2]), min(frame.shape[0], box[1] + box[3])),
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

                # Generate a compliment based on the emotion
                compliment = self.get_compliment(emotion)
                cv2.putText(
                    frame,
                    compliment,
                    (box[0], box[1] + box[3] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

        return frame, faces, emotion_data

    def _load_compliments(self):
        try:
            compliments_path = Path(__file__).parent.parent / 'data' / 'compliments.json'
            with open(compliments_path, 'r') as f:
                compliments = json.load(f)
                logging.info(f"Compliments loaded: {compliments}")
                return compliments
        except FileNotFoundError:
            logging.error("Compliments file not found; using default compliments.")
            return {"default": ["You look wonderful today!"]}
        except Exception as e:
            logging.error(f"Error loading compliments: {str(e)}")
            return {}

    def get_compliment(self, emotion):
        try:
            available_compliments = self.compliments.get(
                emotion,
                self.compliments.get('default', ['You look wonderful today!'])
            )

            unused_compliments = [
                c for c in available_compliments
                if c not in self.used_compliments
            ]

            if not unused_compliments:
                self.used_compliments.clear()
                unused_compliments = available_compliments

            compliment = random.choice(unused_compliments)
            self.used_compliments.add(compliment)

            logging.info(f"Compliment for {emotion}: {compliment}")
            return compliment
        except Exception as e:
            logging.error(f"Error generating compliment: {str(e)}")
            return "You look wonderful today!"

    def run(self):
        self.start_video_capture()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logging.warning("Empty frame received; skipping.")
                    continue
                
                frame, faces, emotion_data = self.process_frame(frame)
                
                cv2.imshow('Smart Detection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            logging.error(f"Runtime error: {str(e)}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        if hasattr(self, 'cap') and self.cap is not None:
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
