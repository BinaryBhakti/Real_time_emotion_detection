import cv2
import numpy as np
from fer import FER
import logging
from datetime import datetime
from pathlib import Path
import random
import json
import pyttsx3

class SmartDetector:
    def __init__(self):
        """Initialize the SmartDetector with face detection, emotion recognition, and TTS capabilities."""
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Initialize emotion detection
        self.emotion_detector = FER(mtcnn=True)
        self.emotion_threshold = 0.5
        
        # Set up logging
        logging.basicConfig(
            filename=f'smart_detector_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Load compliments and initialize tracking
        self.compliments = self._load_compliments()
        self.used_compliments = set()

        # Initialize text-to-speech
        try:
            self.tts_engine = pyttsx3.init()
            logging.info("Text-to-speech engine initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing TTS engine: {str(e)}")
            self.tts_engine = None 

        self.cap = None

    def _load_compliments(self):
        """Load compliments from JSON file or create default if not exists."""
        default_compliments = {
            "happy": [
                "Your smile lights up the room!",
                "That joy looks amazing on you!",
                "Your happiness is contagious!"
            ],
            "sad": [
                "You've got this, keep going!",
                "Tomorrow will be brighter!",
                "Your strength shines through even in tough moments"
            ],
            "angry": [
                "Your passion is powerful!",
                "You handle challenges with grace",
                "Taking deep breaths can help center you"
            ],
            "neutral": [
                "Your presence is calm and confident!",
                "You have such a composed demeanor!",
                "Your poise is admirable"
            ],
            "surprised": [
                "Your expressions are so genuine!",
                "You bring such energy to every moment!",
                "Your reactions make life more interesting"
            ],
            "fear": [
                "You're braver than you know!",
                "You have the courage to face anything",
                "Taking small steps forward is still progress"
            ],
            "disgust": [
                "Your honesty is refreshing!",
                "You have great instincts",
                "Trust your judgment"
            ]
        }

        try:
            compliments_path = Path(__file__).parent / 'data' / 'compliments.json'
            logging.info(f"Attempting to load compliments from: {compliments_path}")

            if not compliments_path.is_file():
                logging.warning(f"Compliments file not found at: {compliments_path}")
                logging.info("Creating data directory and default compliments file...")
                
                data_dir = Path(__file__).parent / 'data'
                data_dir.mkdir(exist_ok=True)
                
                with open(compliments_path, 'w') as f:
                    json.dump(default_compliments, f, indent=4)
                logging.info(f"Created default compliments file at: {compliments_path}")
                return default_compliments

            with open(compliments_path, 'r') as f:
                compliments = json.load(f)
                logging.info("Compliments successfully loaded from file.")
                return compliments
                
        except Exception as e:
            logging.error(f"Error loading compliments: {str(e)}. Using default compliments.")
            return default_compliments

    def start_video_capture(self):
        """Initialize video capture from default camera."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logging.error("Failed to open video capture")
            raise RuntimeError("Could not start video capture")
        logging.info("Video capture started successfully")

    def detect_faces(self, frame):
        """Detect faces in the given frame using Haar cascade classifier."""
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
        """Detect emotions in the given frame using FER."""
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

    def get_compliment(self, emotion):
        """Get a random unused compliment for the given emotion."""
        try:
            available_compliments = self.compliments.get(
                emotion,
                self.compliments.get('neutral', ['You look wonderful today!'])
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

    def speak_compliment(self, compliment):
        """Speak the given compliment using text-to-speech."""
        if self.tts_engine:
            try:
                self.tts_engine.say(compliment)
                self.tts_engine.runAndWait()
            except Exception as e:
                logging.error(f"Error speaking compliment: {str(e)}")
        else:
            logging.warning("Text-to-speech engine is not available.")

    def process_frame(self, frame):
        """Process a single frame: detect faces, emotions, and generate compliments."""
        faces = self.detect_faces(frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        emotion_data = self.detect_emotion(frame)
        compliment = "You look wonderful today!"  # Default compliment

        if emotion_data:
            box = emotion_data['box']
            emotion = emotion_data['emotion']
            probability = emotion_data['probability']

            if box and len(box) == 4:
                # Draw emotion detection box
                cv2.rectangle(
                    frame,
                    (max(0, box[0]), max(0, box[1])),
                    (min(frame.shape[1], box[0] + box[2]), min(frame.shape[0], box[1] + box[3])),
                    (0, 255, 0),
                    2
                )

                # Display emotion and probability
                cv2.putText(
                    frame,
                    f"{emotion}: {probability:.2f}",
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                # Generate and display compliment
                if emotion not in self.used_compliments:
                    compliment = self.get_compliment(emotion)
                    self.speak_compliment(compliment)
                    self.used_compliments.add(emotion)

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

    def run(self):
        """Main loop for running the smart detector."""
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
        """Clean up resources."""
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