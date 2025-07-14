# 🎯 Object Detection and Tracking in Sports Videos

This project detects and tracks players, referees, and a single ball in a football match video using YOLOv8 and SORT tracking.

## 🔧 Setup Instructions

1. **Install dependencies** (Python 3.8+):
   ```bash
   pip install ultralytics opencv-python numpy
Run the script:

bash
Copy
Edit
python detect_from_video.py
Input: A football match video (e.g., sample_video.mp4)
Output: Video window with tracked players, referees, and one ball (press q to quit)

🧠 Notes
Ensures only one ball is tracked.

Keeps consistent IDs for players across frames.

✅ Requirements
ultralytics

opencv-python

numpy

Install all using the command

