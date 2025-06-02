# Lane_Detection_System
# Title: 
Lane Detection for Self-Driving Cars Using Classical DIP Techniques with Real
Time Parameter Control 
# 1. Introduction & Problem Statement 
Lane detection is a crucial component of self-driving car systems and advanced driver-assistance 
systems (ADAS). Accurate and reliable identification of road lanes enables vehicles to navigate 
safely, maintain lane discipline, and anticipate road structure changes. 
This project implements a lane detection system specifically tailored for self-driving car 
scenarios, exclusively using classical Digital Image Processing (DIP) techniques — avoiding deep 
learning models by design. It utilizes edge detection (Canny), Hough Line Transform, and region 
of interest (ROI) masking to identify lane markers in real-world road images. The system 
incorporates a graphical user interface (GUI) for runtime parameter adjustment, allowing users 
to intuitively explore the effects of different image processing parameters on detection 
accuracy. 
# 2. Techniques Used and Justification 
The following classical DIP techniques were applied: 
# Edge Detection (Canny Edge Detector) 
Category: Segmentation 
Justification: 
The Canny edge detector identifies potential lane boundaries by detecting areas of rapid 
intensity change. It provides a clear, noise-reduced binary edge map, essential for reliable lane 
extraction in diverse road and lighting conditions. 
# Region of Interest (ROI) Masking 
Category: Segmentation & Pre-processing 
Justification: 
Since lane lines are typically located in specific areas of an image (usually the lower half and 
converging toward the horizon), a polygon-shaped ROI mask isolates this area, removing 
irrelevant regions like the sky, roadside objects, and adjacent vehicles. This improves detection 
accuracy and computational efficiency. 
# Line Detection (Hough Line Transform) 
Category: Feature Extraction 
Justification: 
The Hough Line Transform is used to identify straight-line segments corresponding to lane lines 
from the edge-detected image. It’s particularly robust against noise and partial occlusions — 
common in real-world road scenes. 
# Filtering (Gaussian Blur) 
Category: Noise Reduction 
Justification: 
A Gaussian blur filter is applied prior to edge detection to smoothen the image, reduce noise, 
and prevent false-positive edge detections. It ensures better continuity of lane markings in the 
presence of road texture or minor artifacts. 
# 3. Results & Analysis 
The system was evaluated on a custom dataset of 20+ real-world road images under varied 
conditions: sunny, cloudy, and dusk. The GUI enabled dynamic control over parameters, offering 
users a way to immediately observe the effect of changing thresholds and mask regions on 
detection results. 
# Displayed Outputs in GUI: 
• Original Image 
• Lane Overlay Image (original image + detected lane lines) 
• Binary Output Image (showing lane lines alone) 
# Performance Highlights: 
• Effective lane detection achieved in both clear and moderately noisy images. 
• Parameter tuning through GUI allowed real-time adaptability for different lighting and 
road types. 
• Colored line overlays differentiated steep slopes from mild ones — aiding in debugging 
and clarity. 
• The system maintained real-time image processing speeds suitable for interactive 
testing. 
# GUI Functionality: 
• Adjustable sliders for: 
o Canny edge detection thresholds 
o Gaussian kernel size 
o Hough transform threshold, minimum line length, and maximum gap 
o ROI mask height 
This runtime modification capability allowed fine-grained control, making the system 
educational for users to experience firsthand how classical DIP parameters influence lane 
detection outcomes.
