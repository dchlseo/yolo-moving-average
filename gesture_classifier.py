import os
import re
from src.moving_average import MovingAverageFilter, class_inference_and_smoothing
from src.video_capture import extract_frames_from_videos, save_extracted_frames
from src.inference_visualization import plot_timeseries, annotate_image, create_composite_image, plot_timeseries_per_frame, create_video_visualization
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

## UNCOMMENT BELOW IF YOU WISH TO SAVE THE FRAMES.
# # Save the extracted frames as images
# save_extracted_frames(frames_dict, output_dir)

# # Generate a list of saved frame filenames
# image_file_names = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]

# # Sort image files if needed (assuming they are named in a sortable manner)
# image_file_names.sort()

# print(f'Imported {len(image_file_names)} files.')

# Import model
model = YOLO('model/weights/best.pt')
names = model.model.names
print('class names: ', names)

# Instantiate the moving average filter
ma_filter = MovingAverageFilter(window_size=10)

# Cache for storing results
cached_results = {}

# Check if using saved frame images or direct frame data
using_saved_images = 'image_file_names' in locals()

if using_saved_images:
    for image_file in image_file_names:
        image_path = os.path.join(output_dir, image_file)
        result = class_inference_and_smoothing(image_path, model, ma_filter)
        cached_results[image_file] = result
else:
    # Perform model inference directly on extracted frames
    for video_path, frames in frames_dict.items():
        for idx, frame in enumerate(frames):
            result = class_inference_and_smoothing(frame, model, ma_filter)
            cached_results[f'{os.path.basename(video_path)}_frame_{idx:03d}'] = result


# # Plotting results
# # Class labels and colors (modify as needed)
# class_labels = {
#     '0': 'Release',
#     '1': 'Folding',
#     '2': 'Grab',
#     '-1': 'Undetected',
#     'None': 'Uncertain'
# }

# class_colors = {
#     'Release': (0, 255, 0),
#     'Folding': (0, 165, 255),
#     'Grab': (255, 0, 0),
#     'Undetected': (0, 0, 255),
#     'Uncertain': (255, 255, 0)
# }

# # Check if using saved frame images or direct frame data
# using_saved_images = 'image_file_names' in locals()

# if using_saved_images:
#     # If using saved images, image_dir should point to the directory containing these images
#     image_dir = output_dir
# else:
#     # If not using saved images, generate annotated images directly from frames
#     for video_path, frames in frames_dict.items():
#         for idx, frame in enumerate(frames):
#             detection = cached_results[f'{os.path.basename(video_path)}_frame_{idx:03d}']
#             annotated_image = annotate_image(frame, detection, class_labels, class_colors)
#             frames_dict[video_path][idx] = annotated_image

#     # Set image_dir to None as images are not read from a directory
#     image_dir = None

# # Determine the total number of frames
# total_frames = len(cached_results)

# # Call the function to create video visualization
# create_video_visualization(cached_results, frames_dict, image_dir, class_labels, class_colors, total_frames=num_frames_per_video, output_video_path='output/visualization_output.mp4', fps=30, type='both')
