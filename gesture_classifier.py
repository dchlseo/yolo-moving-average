import os
import re
from src.video_capture import extract_frames_from_videos
from ultralytics import YOLO


# Define the path to the input video and output directory
data_dir = 'data/input/sample_data.mov'
output_dir = 'data/output'

# Ensure that the output directory exists
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the number of frames to extract from each video
num_frames_per_video = 282

# Run the frame extraction
video_paths = [data_dir]
frames_dict = extract_frames_from_videos(video_paths, num_frames_per_video)

# Check if the number of frames extracted matches the parameter value
for video_path, frames in frames_dict.items():
    print(f"Extracted {len(frames)} frames from {video_path}")
    if len(frames) != num_frames_per_video:
        print(f"Warning: Number of frames extracted ({len(frames)}) does not match the specified value ({num_frames_per_video}).")

# Import model
model = YOLO('model/weights/best.pt')
names = model.model.names
print('class names: ', names)









# # Function to extract number from filename for sorting
# def extract_number(filename):
#     match = re.search(r'(\d+)', filename)
#     return int(match.group(0)) if match else 0

# image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.jpg')], key=extract_number)

# print(f'imported {len(image_files)} files.')

