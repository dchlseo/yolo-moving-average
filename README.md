# Enhancing Frame-Level Stability in YOLO Image Classification with Simple Moving Average for Hand Gesture Recognition

- Also see (Korean ver.):  [Google Slide Presentation (한국어 발표 자료)](https://docs.google.com/presentation/d/15e_nBQsfDuISFk8tTSFhq8_cxQKo__2_M940YDAwWbA/edit#slide=id.p)

## Overview
This repository contains the implementation of a hand gesture classification system using YOLO v8 Nano. The project aims to enhance the frame-level stability in image classification by applying a simple moving average (SMA) technique to smooth out oscillations in the classification of hand gestures.

## Features
- **Object Classification**: Utilizes YOLO v8 Nano for efficient and fast hand gesture recognition.
- **Oscillation Reduction**: Implements SMA to reduce frame-to-frame classification fluctuations.
- **Three-Class System**: Classifies hand gestures into 'open hand', 'folding hand', and 'folded hand'.

## Program Demo
### BEFORE APPLYING SMA (blue line):
[![Demo Video1](assets/composite_original_thumb.png)](assets/composite_original.mov "Watch Demo!")

### AFTER APPLYING SMA (orange line):
[![Demo Video2](assets/composite_both_thumb.png)](assets/composite_both.mov "Click to Watch Demo!")

### Key Highlights of the Project:

- **Frame-Level Oscillation Reduction**: Employs a simple moving average (SMA) strategy to mitigate frame-to-frame oscillations, leading to more stable and consistent gesture recognition.

- **Enhanced Accuracy and Stability**: The SMA technique significantly improves the accuracy and stability of gesture classification, especially in scenarios where traditional methods might struggle with rapid class fluctuations.

### Potential Use Cases:

- **Interactive Virtual/Augmented Reality**: Can be integrated into VR/AR systems for more natural and intuitive hand gesture controls, enhancing user experience in virtual environments.

- **Smart Home Control**: Suitable for gesture-based control systems in smart homes, allowing users to manage devices or settings with simple hand movements.

- **Accessible Computer Interfaces**: Offers an alternative way for individuals with disabilities to interact with computers and digital devices using hand gestures.

- **Educational Tools**: Can be used in educational software, especially in interactive learning environments, to make educational content more engaging and interactive.

- **Gesture-Controlled Robotics**: Applicable in controlling robots or automated systems where hand gestures can be used for commands, making the interaction more intuitive and user-friendly.

These highlights and use cases illustrate the versatility and practicality of your project, emphasizing its potential impact in various technology sectors.

-----------------
## Requirements
- Python 3.9
- Dependencies listed in `requirements.txt`

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/dchlseo/yolo-moving-average.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
## Preparation
- The video(.mov) file should be inside the data/input directory.   

## Usage
To run the gesture classification:
```
python gesture_classifier.py
```

## Visualization
- This step would involve the `inference_visualization.py` module, to:
   - generate a graph image of the label classification (as seen in the demo)
   - save a movie version of the visualization.
```
(TO BE UPDATED...))
```


This README provides a concise yet comprehensive guide to setting up and using the project. Modify it as necessary to fit the specific details and requirements of your implementation.
