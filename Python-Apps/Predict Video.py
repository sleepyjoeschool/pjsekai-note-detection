from ultralytics import YOLO
import cv2
import os

def process_video(video_path="video.mp4", output_path="output.mp4", model_name="model.pt"):
    if not os.path.exists(video_path):
        print(f"[ERROR] The video file does not find in {video_path}")
        return
    
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"The video is now proceed: {video_path}")
    print(f"FPS: {fps:.2f} FPS, Pixel={width}x{height}, Number of Frames={total_frames}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:
            progress = (frame_count / total_frames) * 100
            print(f" {frame_count}/{total_frames} frames has been proceed ({progress:.1f}%)")

        results = model(frame, conf=0.1)

        for result in results:
            annotated_frame = result.plot()

        out.write(annotated_frame)

    cap.release()
    out.release()
    
    print(f"The video is now stored in: {output_path}")

if __name__ == "__main__":
    process_video()