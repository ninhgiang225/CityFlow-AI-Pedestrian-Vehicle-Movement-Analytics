import cv2
import os

def extract_evaluation_frames(video_path, output_dir, num_frames=50):
    """Extract evenly spaced frames for evaluation"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = total_frames // num_frames
    
    saved = 0
    for i in range(num_frames):
        frame_idx = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            cv2.imwrite(f"{output_dir}/frame_{saved:04d}.jpg", frame)
            saved += 1
    
    cap.release()
    print(f"Extracted {saved} frames to {output_dir}")

# Extract 50-100 frames for evaluation
