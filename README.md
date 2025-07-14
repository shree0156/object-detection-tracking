Object Detection & Tracking in Football Matches
This project performs real-time object detection and multi-object tracking for football match videos. It detects players, referees, and the football , and keeps consistent IDs across frames. It uses YOLOv8 for detection and SORT (Simple Online Realtime Tracker) for tracking.

Project Structure
bash
Copy
Edit
.
├── detect_from_video.py          # Main script for running detection + tracking
├── sort.py                       # SORT tracking logic
├── best.pt                       # Trained YOLOv8 model
├── sample_video.mp4              # Example input video
├── README.md                     # Project documentation
└── Object_Detection_Tracking_Report.pdf  # Final report
Requirements & Setup
All required packages are listed below. You don't need a separate requirements.txt.

Dependencies:
Python 3.8+

Ultralytics (pip install ultralytics)

OpenCV (pip install opencv-python)

NumPy (pip install numpy)

🔧 Installation Steps:
Clone the repository

bash
Copy
Edit
git clone https://github.com/shree0156/object-detection-tracking.git
cd object-detection-tracking
Install dependencies
(Use a virtual environment if you prefer)

nginx
Copy
Edit
pip install ultralytics opencv-python numpy
Place your input video
Replace sample_video.mp4 with your own football match video if needed.

Run the detection & tracking script

nginx
Copy
Edit
python detect_from_video.py
This will:

Detect players, referees, and the ball.

Assign unique and consistent IDs.

Show a live video window with bounding boxes and IDs.

Notes
The model used is a fine-tuned YOLOv8 trained to detect 3 classes: player, referee, and ball.

SORT has been adapted to keep only one ball in detection even if multiple are found.

Player tracking is consistent across frames using unique IDs.

Output
Live video preview with tracked objects.

Optionally, you can modify the code to save output video (using cv2.VideoWriter).

Report
The project report explaining the approach, experiments, challenges, and future improvements is available as Object_Detection_Tracking_Report.pdf in this repository.

Credits
Ultralytics YOLOv8

SORT Tracker



numpy

Install all using the command

