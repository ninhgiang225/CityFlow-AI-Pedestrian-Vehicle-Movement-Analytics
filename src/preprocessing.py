import cv2
import numpy as np

def preprocess_video(input_path, output_path, target_fps=30, target_width=1280):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise Exception("Cannot open input video")

    # original fps
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    print("Original FPS:", orig_fps)

    target_height = int(target_width * 9 / 16)  # 16:9 ratio

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (target_width, target_height))

    last_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to standard 1280x720
        frame = cv2.resize(frame, (target_width, target_height))

        # Remove duplicate frames (difference threshold)
        if last_frame is not None:
            diff = cv2.absdiff(frame, last_frame)
            if np.mean(diff) < 2:   # threshold
                continue

        # Write frame
        out.write(frame)
        last_frame = frame

    cap.release()
    out.release()
    print("Saved:", output_path)



