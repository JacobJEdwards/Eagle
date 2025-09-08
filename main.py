import os

import cv2
import numpy as np
from argparse import ArgumentParser
from eagle.models.coordinate_model import CoordinateModel
from eagle.processor import RealTimeProcessor

def annotate_frame(frame, frame_data, team_mapping):
    """
    Annotates a single frame with the detected objects and keypoints.
    """
    if not frame_data:
        return frame

    # Draw Player and Goalkeeper annotations
    for entity_type in ["Player", "Goalkeeper"]:
        if entity_type in frame_data["Coordinates"]:
            for player_id, data in frame_data["Coordinates"][entity_type].items():
                x, y = data["Bottom_center"]
                team_id = team_mapping.get(player_id)

                if team_id == 0:
                    color = (255, 0, 0)  # Blue for team 0
                elif team_id == 1:
                    color = (0, 0, 255)  # Red for team 1
                else:
                    color = (0, 255, 0)  # Green for others/goalkeepers

                cv2.ellipse(frame, (int(x), int(y)), (35, 18), 0, -45, 235, color, 2)
                cv2.putText(frame, str(player_id), (int(x) - 10, int(y) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw Ball annotation
    if "Ball" in frame_data["Coordinates"]:
        for ball_id, data in frame_data["Coordinates"]["Ball"].items():
            x, y = data["Bottom_center"]
            cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 255), -1) # Yellow circle for the ball

    # Draw Keypoints
    if "Keypoints" in frame_data:
        for point in frame_data["Keypoints"].values():
            cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 0), -1)

    return frame


def main_realtime():
    parser = ArgumentParser()
    parser.add_argument("--video_path", type=str, default="0", help="Path to the video file or '0' for webcam.")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for processing.")
    args = parser.parse_args()

    # Initialize the model and real-time processor
    model = CoordinateModel()
    processor = RealTimeProcessor(fps=args.fps)

    video_source = 0 if args.video_path == "0" else args.video_path
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source at {args.video_path}")
        return

    annotated_frames = []

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame {i}")
        i += 1

        # Process a single frame to get coordinates and other data
        frame_data = model.process_single_frame(frame, fps=args.fps)

        if frame_data:
            # Update the real-time processor and get the latest team mapping
            processor.update(frame, frame_data)
            team_mapping = processor.get_team_mapping()

            # Annotate the frame with the processed data
            annotated_frame = annotate_frame(frame, frame_data, team_mapping)
            # cv2.imshow("Eagle Real-Time Tracking", annotated_frame)
            annotated_frames.append(annotated_frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    os.mkdir("output") if not os.path.exists("output") else None
    if annotated_frames:
        for idx, frame in enumerate(annotated_frames):
            cv2.imwrite(f"output/annotated_frame_{idx:04d}.png", frame)

    cap.release()

if __name__ == "__main__":
    main_realtime()