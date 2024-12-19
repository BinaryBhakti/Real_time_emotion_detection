from fer import FER
import cv2
import numpy as np
import logging
from datetime import datetime

class FaceDetector:
    def __init__(self):
        # Initialize face cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if self.face_cascade.empty():
            raise IOError("Failed to load Haar cascade for face detection.")
        
        # Initialize emotion detector
        self.emotion_detector = FER(mtcnn=True)
        
        # Logging setup
        logging.basicConfig(
            filename=f'smart_mirror_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.emotion_threshold = 0.5  # Probability threshold for emotions

    def detect_emotion(self, frame):
        """Detect emotions from a frame."""
        try:
            emotions = self.emotion_detector.detect_emotions(frame)
            if emotions:
                dominant_emotion = max(
                    emotions[0]['emotions'].items(),
                    key=lambda x: x[1]
                )
                if dominant_emotion[1] > self.emotion_threshold:
                    return {
                        'emotion': dominant_emotion[0],
                        'probability': dominant_emotion[1],
                        'box': emotions[0]['box']
                    }
            return None
        except Exception as e:
            logging.error(f"Error in detect_emotion: {e}")
            return None

    def process_frame(self, frame):
        """Detect faces and overlay emotion analysis."""
        # Detect emotion
        emotion_data = self.detect_emotion(frame)

        if emotion_data:
            # Draw bounding box for emotion
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
        return frame

    def start_video_capture(self):
        """Start video capture."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logging.error("Failed to open video capture")
            raise RuntimeError("Could not start video capture")

    def run(self):
        """Main loop for processing video feed."""
        self.start_video_capture()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to grab frame")
                break

            frame = self.process_frame(frame)

            cv2.imshow('Emotion Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        """Release resources."""
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
