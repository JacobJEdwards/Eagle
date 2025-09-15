import base64
import os
import cv2
import numpy as np
from argparse import ArgumentParser
import json
import time
from eagle.models.coordinate_model import CoordinateModel
from eagle.processor import RealTimeProcessor

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            data_b64 = base64.b64encode(obj.tobytes()).decode('utf-8')
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)

        if isinstance(obj, np.generic):
            return obj.item()

        return super().default(obj)

def json_numpy_obj_hook(dct):
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct

def dumps(*args, **kwargs):
    kwargs.setdefault('cls', NumpyEncoder)
    return json.dumps(*args, **kwargs)

def loads(*args, **kwargs):
    kwargs.setdefault('object_hook', json_numpy_obj_hook)
    return json.loads(*args, **kwargs)

def dump(*args, **kwargs):
    kwargs.setdefault('cls', NumpyEncoder)
    return json.dump(*args, **kwargs)

def load(*args, **kwargs):
    kwargs.setdefault('object_hook', json_numpy_obj_hook)
    return json.load(*args, **kwargs)

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
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files.")
    args = parser.parse_args()

    model = CoordinateModel()

    video_source = 0 if args.video_path == "0" else args.video_path
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source at {args.video_path}")
        return

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps == 0:
        native_fps = 30  # Default for webcams

    processor = RealTimeProcessor(fps=native_fps)

    os.makedirs(args.output_dir, exist_ok=True)
    json_output_dir = os.path.join(args.output_dir, "json_data")
    frames_output_dir = os.path.join(args.output_dir, "annotated_frames")
    os.makedirs(json_output_dir, exist_ok=True)
    os.makedirs(frames_output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = None

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        frame_data = model.process_single_frame(frame, fps=native_fps)

        if frame_data:
            processor.update(frame, frame_data)
            team_mapping = processor.get_team_mapping()

            json_filename = os.path.join(json_output_dir, f"frame_{frame_count:04d}.json")
            with open(json_filename, 'w') as f:
                dump(frame_data, f, cls=NumpyEncoder, indent=4)

            annotated_frame = annotate_frame(frame.copy(), frame_data, team_mapping)

            frame_filename = os.path.join(frames_output_dir, f"annotated_frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, annotated_frame)

            if out_video is None:
                h, w, _ = annotated_frame.shape
                out_video = cv2.VideoWriter(os.path.join(args.output_dir, "annotated_video.mp4"), fourcc, native_fps, (w,h))

            out_video.write(annotated_frame)
            cv2.imshow("Eagle Real-Time Tracking", annotated_frame)

        processing_time = time.time() - start_time

        frames_to_skip = int(processing_time * native_fps)

        frame_count += frames_to_skip + 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out_video:
        out_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_realtime()