import cv2

def extract_frames(video_path, output_dir):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if the video file was successfully opened
    if not video.isOpened():
        print("Error opening video file")
        return
    
    # Initialize frame counter
    frame_count = 0
    
    while True:
        # Read the next frame from the video
        ret, frame = video.read()
        
        # If the frame was not successfully read, then we have reached the end of the video
        if not ret:
            break
        
        # Increment frame counter
        frame_count += 1
        
        # Construct the output path
        output_path = f"{output_dir}/frame_{frame_count}.jpg"
        cv2.imwrite(output_path, frame)
            
    # Release the video file
    video.release()
