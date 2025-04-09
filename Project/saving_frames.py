import cv2
import os

def save_frames(video, output_folder = r'C:\Users\stct\Documents\Original Frames', n=5):
    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

    for idx, frame in enumerate(frame_list):
    # Create the file name for each frame
        frame_filename = os.path.join(output_folder, f"frame_{idx+1}.jpg")
        
        # Save the frame as an image in the output folder
        cv2.imwrite(frame_filename, frame)

    return frame_list

if __name__ == "__main__":
    video = 'C:/Users/stct/Downloads/Steinermandl_TopStation_FallDown/Steinermandl_TopStation_FallDown/SteinermandlFallDown_CAM2-1735209480-1735209540.mp4'
    output_folder = r'C:\Users\stct\Documents\Original Frames'
    n = 30 # You can set this to the number of frames you want to extract
    frame_list = save_frames(video, output_folder, n)
    frame_shape = frame_list[1].shape