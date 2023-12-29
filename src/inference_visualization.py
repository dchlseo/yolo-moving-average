import io
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
from PIL import Image

def plot_timeseries(cached_results, type='both'):
    """
    Plots a time series of the original and adjusted class names detected in images.

    Args:
    cached_results (dict): Dictionary of detection results per image.
    type (str): Specifies whether to plot 'original', 'adjusted', or 'both' types of data.
    """
    original_data = []
    adjusted_data = []
    for file, detection in cached_results.items():
        original_class = str(detection['Original Class'])
        adjusted_class = str(detection['Adjusted Class'])

        if type in ['original', 'both']:
            original_data.append({'Image': file, 'Class Name': original_class, 'Type': 'Original'})
        if type in ['adjusted', 'both']:
            adjusted_data.append({'Image': file, 'Class Name': adjusted_class, 'Type': 'Adjusted'})

    combined_data = original_data + adjusted_data
    df = pd.DataFrame(combined_data)
    category_order = ['2', '1', '0', '-1', 'None']
    df['Class Name'] = pd.Categorical(df['Class Name'], categories=category_order, ordered=True)

    plt.figure(figsize=(15, 8))
    sns.lineplot(x='Image', y='Class Name', hue='Type', data=df, marker='o')
    plt.title('Original vs. Adjusted Class Names Detected in Images')
    plt.xlabel('Time')
    plt.ylabel('Class Name')
    plt.tight_layout()
    plt.xticks([])  # Remove x-ticks
    plt.show()


def annotate_image(image, detection, class_labels, class_colors):
    """
    Annotates an image with a label and confidence score.

    Args:
    image (numpy array): The image to annotate.
    detection (dict): Detection result containing class and confidence score.
    class_labels (dict): Dictionary mapping class IDs to labels.
    class_colors (dict): Dictionary mapping class labels to colors.
    """
    adjusted_class = str(detection['Adjusted Class'])
    label = class_labels.get(adjusted_class, 'Unknown')
    conf = detection['Confidence Score']

    font_scale = 1.5
    font_thickness = 4
    text_position = (20, 50)
    text_color = class_colors.get(label, (255, 255, 255))
    label_text = f"{label}: {conf:.2f}" if adjusted_class != '-1' else label

    return cv2.putText(image, label_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)


def create_composite_image(annotated_image, plot_path, output_path):
    """
    Creates a composite image by combining an annotated image with a plot.

    Args:
    annotated_image (numpy array): Annotated image.
    plot_path (str): Path to the saved plot image.
    output_path (str): Path to save the composite image.
    """
    annotated_image_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plot = Image.open(plot_path)

    plot = plot.resize((annotated_image_pil.width, int(annotated_image_pil.width / plot.width * plot.height)))
    composite_image = Image.new('RGB', (annotated_image_pil.width, annotated_image_pil.height + plot.height))

    composite_image.paste(annotated_image_pil, (0, 0))
    composite_image.paste(plot, (0, annotated_image_pil.height))
    composite_image.save(output_path)


def plot_timeseries_per_frame(cached_results, output_dir, image_dir, class_labels, class_colors, total_frames, total_cateogory, type='both'):
    """
    Plots time series and creates composite images for each frame, illustrating the detection results over time.

    Args:
    cached_results (dict): Dictionary of detection results per image.
    output_dir (str): Directory to save the plots and composite images.
    image_dir (str): Directory containing the original images.
    class_labels (dict): Dictionary mapping class IDs to labels.
    class_colors (dict): Dictionary mapping class labels to colors.
    total_frames (int): Total number of frames.
    total_category (int): Total number of categories.
    type (str): Specifies whether to plot 'original', 'adjusted', or 'both' types of data.
    """
    os.makedirs(output_dir, exist_ok=True)

    original_data = []
    adjusted_data = []

    for i, (file, detection) in enumerate(cached_results.items()):
        print(f'plotting: {i+1} / {len(cached_results)}...')
        image_path = os.path.join(image_dir, file)
        image = cv2.imread(image_path)
        annotated_image = annotate_image(image, detection, class_labels, class_colors)

        original_class = str(detection['Original Class'])
        adjusted_class = str(detection['Adjusted Class'])
        if type in ['original', 'both']:
            original_data.append({'Image': file, 'Class Name': original_class, 'Type': 'Original'})
        if type in ['adjusted', 'both']:
            adjusted_data.append({'Image': file, 'Class Name': adjusted_class, 'Type': 'Adjusted'})

        combined_data = original_data + adjusted_data
        df = pd.DataFrame(combined_data)
        category_order = ['None', '-1', '0', '1', '2']
        df['Class Name'] = pd.Categorical(df['Class Name'], categories=category_order, ordered=True)

        plt.figure(figsize=(15, 8))
        sns.lineplot(x='Image', y='Class Name', hue='Type', data=df, marker='o')
        plt.title(f'Original vs. Adjusted Class Names Detected in Images (Up to Frame {i+1})')
        plt.xlabel('Time')
        plt.ylabel('Class Name')
        plt.xlim(0, total_frames)
        plt.ylim(-0.5, len(category_order) - 0.5)
        plt.tight_layout()
        plt.xticks([])  # Remove x-ticks
        plot_path = os.path.join(output_dir, f'plot_{i+1:04d}.png')
        plt.savefig(plot_path)
        plt.close()

        composite_path = os.path.join(output_dir, f'composite_{i+1:04d}.jpg')
        create_composite_image(annotated_image, plot_path, composite_path)
    
    print('COMPLETE.')

def get_plot_as_image(figure):
    """ Convert a Matplotlib figure to a PIL Image and return it """
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img

def create_video_visualization(cached_results, frames_dict, image_dir, class_labels, class_colors, total_frames, output_video_path='output/visualization_output.mp4', fps=30, type='both'):
    """
    Creates a video visualization showing annotated images and corresponding time series plots.

    Args:
    cached_results (dict): Dictionary of detection results per image.
    frames_dict (dict): Dictionary with frames data if image_dir is None.
    image_dir (str or None): Directory containing the original images or None if using frames from frames_dict.
    class_labels (dict): Dictionary mapping class IDs to labels.
    class_colors (dict): Dictionary mapping class labels to colors.
    total_frames (int): Total number of frames.
    fps (int): Frames per second for the output video.
    type (str): Specifies whether to plot 'original', 'adjusted', or 'both' types of data.
    """
    # Initialize lists to store data for plots
    original_data = []
    adjusted_data = []

    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = output_video_path
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    video_writer = None

    for i, (file, detection) in enumerate(cached_results.items()):
        # Handle the case where image_dir is None (using frames directly)
        if image_dir is None:
            image = frames_dict[file]  # Assuming 'file' is the key for frames_dict
        else:
            image_path = os.path.join(image_dir, file)
            image = cv2.imread(image_path)

        # Annotate image
        annotated_image = annotate_image(image, detection, class_labels, class_colors)

        # Convert the annotated OpenCV image to PIL format for consistency
        annotated_image_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

        # Create plot data
        original_class = str(detection['Original Class'])
        adjusted_class = str(detection['Adjusted Class'])
        if type in ['original', 'both']:
            original_data.append({'Image': file, 'Class Name': original_class, 'Type': 'Original'})
        if type in ['adjusted', 'both']:
            adjusted_data.append({'Image': file, 'Class Name': adjusted_class, 'Type': 'Adjusted'})

        combined_data = original_data + adjusted_data
        df = pd.DataFrame(combined_data)

        # Plot
        plt.figure(figsize=(8, 4))  # Adjust the size to match your frame size
        sns.lineplot(x='Image', y='Class Name', hue='Type', data=df, marker='o')
        plt.xticks([])  # Remove x-ticks for clarity
        plt.title(f'Frame {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Class Name')

        # Convert plot to image
        plot_fig = plt.gcf()
        plt.tight_layout()
        plot_img = get_plot_as_image(plot_fig)
        plt.close()

        # Resize plot image to match width of annotated image
        plot_img_resized = plot_img.resize((annotated_image_pil.width, int(annotated_image_pil.width / plot_img.width * plot_img.height)))

        # Combine annotated image and resized plot image
        combined_frame = Image.new('RGB', (annotated_image_pil.width, annotated_image_pil.height + plot_img_resized.height))
        combined_frame.paste(annotated_image_pil, (0, 0))
        combined_frame.paste(plot_img_resized, (0, annotated_image_pil.height))

        # Convert combined frame to OpenCV format for video writing
        combined_frame_cv = cv2.cvtColor(np.array(combined_frame), cv2.COLOR_RGB2BGR)

        # Initialize video writer with the size of combined_frame
        if video_writer is None:
            height, width = combined_frame_cv.shape[:2]
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        video_writer.write(combined_frame_cv)

    video_writer.release()
    print('Video visualization created:', video_path)
