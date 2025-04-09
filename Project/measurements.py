import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import image_quality


def load_frames_path(image_folders, frame_list=[]):
    """Get a list of image file paths from multiple folders."""
    if not isinstance(image_folders, list):
        image_folders = [image_folders]  # If a single folder path is passed, convert to list
    for folder in image_folders:
        if os.path.exists(folder):
            image_files = [f for f in os.listdir(folder) if f.endswith(('jpg', 'png', 'jpeg'))]    # Get a list of all image files in the folder (only .jpg, .png, .jpeg)
            image_paths = [os.path.join(folder, f) for f in image_files]  # Get full path of images
            frame_list.extend(image_paths)  # Add to the existing list
        else:
            print(f"Folder {folder} does not exist.")
    return frame_list  # Return the list of all image paths

def create_data_metric(folders):  # as imput use a list of all folder pat
    frame_list = load_frames_path(image_folders = folders, frame_list = [])
    frame_analysis_list = []  # List to store the results

    # Loop through each frame in the frame_list
    for framepath in frame_list:
        frame = cv2.imread(framepath)
        frame_analysis = image_quality.evaluate_image_quality_metrics(frame)  # Apply the function to each frame
        frame_analysis_list.append(frame_analysis) 

    metric_measure = pd.DataFrame(frame_analysis_list)
    stats_df = metric_measure.agg(['min', 'max', 'var']).transpose()

    return metric_measure, stats_df

if __name__ == "__main__":
    folder = r"C:\Users\stct\Documents\Original Frames"
    folder2 = r"C:\Users\stct\Documents\Modified Frames"
    image_folders = [folder, folder2]
    metric_measure, stats_df = create_data_metric(image_folders)

    print(metric_measure)
    print(stats_df)

    num_columns = len(metric_measure.columns)

    # Create subplots with one row and the number of columns
    fig, axes = plt.subplots(1, num_columns, figsize=(15, 5))

    # If there is only one column, axes will be a single subplot (not a list), so make it iterable
    if num_columns == 1:
        axes = [axes]

    # Loop through each column and plot its histogram
    for idx, column in enumerate(metric_measure.columns):
        axes[idx].hist(metric_measure[column], bins=15, edgecolor='black')
        axes[idx].set_title(f'{column} Distribution')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()
