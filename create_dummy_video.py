import cv2
import numpy as np

def create_dummy_video(filename="dummy_test.mp4", duration=10, fps=30, width=1920, height=1080):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(duration * fps):
        # Create a frame with some motion
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw a moving rectangle
        x = (i * 10) % width
        y = (i * 5) % height
        cv2.rectangle(frame, (x, y), (x + 200, y + 200), (0, 255, 0), -1)
        # Add some text
        cv2.putText(frame, f"Frame {i}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        out.write(frame)

    out.release()
    print(f"Dummy video created: {filename}")

if __name__ == "__main__":
    create_dummy_video()
