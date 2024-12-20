# Real Time Emotion Detection System

## Overview
This Real time emotion Detection System is an interactive application that combines facial detection and emotion recognition capabilities. The system can detect faces in real-time using a webcam, analyze emotions, and provide visual feedback. It's built using Python and leverages computer vision and machine learning technologies.

## Features
- Real-time face detection using OpenCV
- Emotion recognition using FER (Facial Emotion Recognition)
- Visual feedback with bounding boxes and emotion labels
- Comprehensive logging system
- Error handling and graceful degradation
- Performance-optimized processing pipeline

## Requirements

### Hardware
- Webcam or USB camera
- Computer with Python support

### Software Dependencies
```bash
Python 3.7+
OpenCV (cv2)
NumPy
FER (Facial Emotion Recognition)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/smart-mirror.git
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python src/face_detector.py
```

2. Controls:
- Press 'q' to quit the application
- The application will automatically detect faces and emotions when they appear in frame

<!-- ## Project Structure
```
smart-mirror/
├── src/
│   ├── smart_detector.py    # Main detection system
│   └── __init__.py
├── logs/                    # Log files directory
├── tests/                   # Test files
├── requirements.txt         # Project dependencies
└── README.md               # This file
``` -->

## Detection System

### Face Detection
- Uses OpenCV's Haar Cascade Classifier
- Displays blue bounding boxes around detected faces
- Optimized for real-time processing

### Emotion Detection
- Utilizes the FER library with MTCNN
- Displays green bounding boxes with emotion labels
- Shows emotion probability scores
- Supports multiple emotion categories

## Error Handling
- Comprehensive logging system
- Graceful fallback mechanisms
- Runtime error recovery
- Resource cleanup on exit

## Logging
The system maintains detailed logs including:
- Detection events
- System errors
- Runtime performance
- Resource management

Log files are created with timestamps in the format: `smart_detector_YYYYMMDD.log`

## Performance Considerations
- Face detection runs first (faster processing)
- Emotion detection processes whole frame
- Optimized frame processing pipeline
- Resource-conscious design

## Development

## Future Enhancements
- Multiple face tracking
- Emotion history tracking
- Configuration UI
- Performance optimizations
- Additional emotion categories
- Custom detection parameters

## Troubleshooting

### Common Issues

1. Camera not found:
```
Error: Could not start video capture
Solution: Check camera connections and permissions
```

2. Dependencies missing:
```
Solution: Run 'pip install -r requirements.txt'
```

3. Performance issues:
```
Solution: Adjust frame processing parameters in smart_detector.py
```

4. Compliment not loaded:
```
Solution: Attached to the correct file location
```

5. Text to speech:
```
Solution: Tried to modify it with better solution, still there are some issues.
```


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- OpenCV team for computer vision tools
- FER library developers
- Contributors and testers