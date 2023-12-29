import cv2
import os
import glob

def extract_frames_from_videos(video_paths, num_frames_per_video=300):
    """Extract a specified number of frames from multiple videos.
    Args:
        video_paths (list): List of video file paths
        num_frames_per_video (int): Number of frames to extract from each video
    Returns:
        dict: Dictionary with each video path as key and a list of extracted frames as value
    """
    frames_dict = {}

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Unable to open video file: {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = total_frames // (num_frames_per_video - 1)
        frames = []

        for i in range(0, total_frames, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if len(frames) % 10 == 0:
                print(f"Processing frame {len(frames)} / {num_frames_per_video} from video: {video_path}")
            if len(frames) == num_frames_per_video:
                break

        cap.release()
        frames_dict[video_path] = frames

    return frames_dict

def save_extracted_frames(frames_dict, base_save_path):
    """Save the extracted frames as image files.
    Args:
        frames_dict (dict): Dictionary with video paths as keys and lists of frames as values
        base_save_path (str): Base path for saving
    """
    for video_path, frames in frames_dict.items():
        video_name = os.path.basename(video_path).split('.')[0]
        save_path = os.path.join(base_save_path, video_name)
        os.makedirs(save_path, exist_ok=True)

        for i, frame in enumerate(frames):
            frame_number = f'{i:03d}'  # Format the frame number to be three digits
            frame_path = os.path.join(save_path, f'frame_{frame_number}.jpg')
            cv2.imwrite(frame_path, frame)
