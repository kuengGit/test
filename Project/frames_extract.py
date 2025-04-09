import cv2
import pandas as pd
import image_quality
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('C:/Users/stct/Downloads/Steinermandl_TopStation_FallDown/Steinermandl_TopStation_FallDown/SteinermandlFallDown_CAM2-1735209480-1735209540.mp4')
n = 10  # You can set this to the number of frames you want to extract

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate the interval between frames (frame number step)
frame_interval = total_frames // n


frame_list = []

# Extract frames and store them in the list
for i in range(n):
    # Set the frame position
    frame_pos = i * frame_interval
    
    # Set the video capture object to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    
    # Read the frame at the current position
    ret, frame = cap.read()
    
    if ret:
        # Store the frame in the list
        frame_list.append(frame)
    else:
        print(f"Error reading frame at position {frame_pos}")

# Release the video capture object
cap.release()


#Analyse metrics
frame_analysis_list = []  # List to store the results


# Loop through each frame in the frame_list
for frame in frame_list:
    frame_analysis = image_quality.evaluate_image_quality_metrics(frame)  # Apply the function to each frame
    frame_analysis_list.append(frame_analysis) 

metric_measure = pd.DataFrame(frame_analysis_list)

print(metric_measure)

metric_measure['Black Clipping'].plot(kind='hist', bins=10, edgecolor='black')
plt.show()

stats_df = metric_measure.agg(['min', 'max', 'var']).transpose()

# Display the stats DataFrame
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

