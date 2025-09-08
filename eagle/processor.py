from sklearn.cluster import KMeans
import pandas as pd
import cv2
import numpy as np
import math
from collections import Counter, deque

PITCH_WIDTH = 105
PITCH_HEIGHT = 68
color_ranges = {
    "red": [(0, 100, 100), (10, 255, 255)],
    "red2": [(160, 100, 100), (179, 255, 255)],
    "orange": [(11, 100, 100), (25, 255, 255)],
    "yellow": [(26, 100, 100), (35, 255, 255)],
    "green": [(36, 100, 100), (85, 255, 255)],
    "cyan": [(86, 100, 100), (95, 255, 255)],
    "blue": [(96, 100, 100), (125, 255, 255)],
    "purple": [(126, 100, 100), (145, 255, 255)],
    "magenta": [(146, 100, 100), (159, 255, 255)],
    "white": [(0, 0, 200), (180, 30, 255)],
    "gray": [(0, 0, 50), (180, 30, 200)],
    "black": [(0, 0, 0), (180, 255, 50)],
}


def calculate_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def interpolate_df(df, col_name: str, fill: bool = False):
    # Work on temporary Series to avoid DataFrame fragmentation
    s = df[col_name]
    x = s.apply(lambda v: v[0] if isinstance(v, (list, tuple)) else np.nan)
    y = s.apply(lambda v: v[1] if isinstance(v, (list, tuple)) else np.nan)

    if fill:
        x = x.interpolate(method="linear").bfill().ffill()
        y = y.interpolate(method="linear").bfill().ffill()
    else:
        x = x.interpolate(method="linear", limit_area="inside")
        y = y.interpolate(method="linear", limit_area="inside")

    combined = pd.Series([(xi, yi) if not (math.isnan(xi) and math.isnan(yi)) else np.nan for xi, yi in zip(x, y)], index=s.index)
    df[col_name] = combined
    return df


def smooth_df(df, col_name: str):
    # Work on temporary Series to avoid DataFrame fragmentation
    s = df[col_name]
    x = s.apply(lambda v: v[0] if isinstance(v, (list, tuple)) else np.nan)
    y = s.apply(lambda v: v[1] if isinstance(v, (list, tuple)) else np.nan)

    x.iloc[::2] = np.nan
    y.iloc[::2] = np.nan
    x = x.interpolate(method="linear", limit_area="inside")
    y = y.interpolate(method="linear", limit_area="inside")

    combined = pd.Series([(xi, yi) if not (math.isnan(xi) and math.isnan(yi)) else np.nan for xi, yi in zip(x, y)], index=s.index)
    df[col_name] = combined
    return df


class Processor:
    def __init__(self, coords, frames: list, fps: int, debug: bool = False, filter_ball_detections: bool = False):
        assert len(coords) == len(frames), f"Length of coords ({len(coords)}) and frames ({len(frames)}) should be the same"
        self.coords = coords  # Data should be same format as CoordinateModel output
        self.frames = frames
        self.fps = fps
        self.debug = debug
        self.filter_ball_detections = filter_ball_detections

    def process_data(self, smooth: bool = False):
        df = self.create_dataframe()
        if df.empty:
            return df, {}
        df = interpolate_df(df, "Ball", fill=True)
        df = interpolate_df(df, "Ball_video", fill=True)
        team_mapping = self.get_team_mapping()
        df.index = df.index.astype(int)
        df = self.merge_data(df, team_mapping)  # this is to fix any tracker errors (the same player is given different ids in different frames)

        for col in df.columns:
            df = interpolate_df(df, col, fill=False)
            if smooth:
                df = smooth_df(df, col)
        return df, team_mapping

    def format_data(self, df):
        out = []
        for frame_number in df.index:
            indiv = {}
            indiv["Boundaries"] = [df.loc[frame_number, "Bottom_Left"], df.loc[frame_number, "Top_Left"], df.loc[frame_number, "Top_Right"], df.loc[frame_number, "Bottom_Right"]]
            row = df.loc[frame_number]
            indiv_data = []
            indiv_data_video = []
            # select non na values
            for col in df.columns:
                if col in ["Bottom_Left", "Top_Left", "Top_Right", "Bottom_Right"]:
                    continue
                val = row[col]
                if pd.isna(val):
                    continue
                if "ball" in col.lower():  # handle separately
                    continue
                id = col.split("_")[1]
                id = int(id)
                col_type = col.split("_")[0]
                item = {"ID": id, "Coordinates": val, "Type": col_type}
                if "video" in col:
                    indiv_data_video.append(item)
                else:
                    indiv_data.append(item)

            # handle ball
            ball = row["Ball"]
            indiv_data.append({"ID": "Ball", "Coordinates": ball})
            ball_video = row["Ball_video"]
            indiv_data_video.append({"ID": "Ball", "Coordinates": ball_video})

            indiv["Coordinates"] = indiv_data
            indiv["Coordinates_video"] = indiv_data_video

            out.append(indiv)
        return pd.DataFrame(out)

    def create_dataframe(self):
        # We will compute ball positions across ALL frames for stability,
        # but only keep frames that have at least one detected Player or Goalkeeper.
        ball_coords_image_all = []
        ball_coords_all = []
        out = {}
        kept_frame_numbers = []
        frame_keys = list(self.coords.keys())

        for frame_number in frame_keys:
            indiv = {}
            curr = self.coords[frame_number]
            boundaries = curr["Boundaries"]
            indiv["Bottom_Left"] = boundaries[0]
            indiv["Top_Left"] = boundaries[1]
            indiv["Top_Right"] = boundaries[2]
            indiv["Bottom_Right"] = boundaries[3]

            has_person_detection = False
            coordinates_dict = curr.get("Coordinates", {})

            # Add Player and Goalkeeper detections if present
            for name in ["Player", "Goalkeeper"]:
                if name not in coordinates_dict or len(coordinates_dict[name]) == 0:
                    continue
                curr_coords = coordinates_dict[name]
                for id, item in curr_coords.items():
                    x1, y1, x2, y2 = item["BBox"]
                    indiv[f"{name}_{id}"] = item.get("Transformed_Coordinates") if item.get("Transformed_Coordinates") else np.nan
                    indiv[f"{name}_{id}_video"] = ((x1 + x2) / 2, y2)
                    has_person_detection = True

            # Ball candidates (computed for all frames)
            if "Ball" in coordinates_dict and len(coordinates_dict["Ball"]) > 0:
                curr_coords = coordinates_dict["Ball"]
                indiv_img = []
                indiv_real = []
                for id, item in curr_coords.items():
                    confidence = float(item["Confidence"])
                    transformed_coords = item["Transformed_Coordinates"]
                    x1, y1, x2, y2 = item["BBox"]
                    center = ((x1 + x2) / 2, y2)
                    if not transformed_coords:
                        transformed_coords = center
                    indiv_real.append((transformed_coords, confidence))
                    indiv_img.append((center, confidence))

                indiv_img = sorted(indiv_img, key=lambda x: x[1], reverse=True)
                indiv_real = sorted(indiv_real, key=lambda x: x[1], reverse=True)
                ball_coords_all.append([x[0] for x in indiv_real])
                ball_coords_image_all.append([x[0] for x in indiv_img])
            else:
                ball_coords_all.append(None)
                ball_coords_image_all.append(None)

            # Only keep frames with at least one player/goalkeeper detection
            if has_person_detection:
                out[frame_number] = indiv
                kept_frame_numbers.append(frame_number)

        # Compute final ball positions on the full timeline, then align to kept frames
        h, w, _ = self.frames[0].shape
        final_ball_coords_img_all = self.parse_ball_detections_with_kalman(ball_coords_image_all, filter=self.filter_ball_detections, threshold=0.1 * w)
        final_ball_coords_all = self.parse_ball_detections_with_kalman(ball_coords_all, filter=False)
        # Use image coordinates to filter real-world coordinates
        final_ball_coords_all = [final_ball_coords_all[i] if final_ball_coords_img_all[i] is not None else None for i in range(len(final_ball_coords_img_all))]

        df = pd.DataFrame(out).T
        if len(df) > 0:
            # Align ball series to kept frame indices
            ball_series_real = pd.Series([x if x is not None else np.nan for x in final_ball_coords_all], index=frame_keys)
            ball_series_img = pd.Series([x if x is not None else np.nan for x in final_ball_coords_img_all], index=frame_keys)
            df["Ball"] = ball_series_real.loc[df.index]
            df["Ball_video"] = ball_series_img.loc[df.index]
            # Remove columns with less than 1% non-None values
            df = df.loc[:, df.notna().sum() >= 0.01 * len(df)]
        return df

    def merge_data(self, df, team_mapping):
        goal_keeper_cols = [x for x in df.columns if "Goalkeeper" in x and "video" in x]
        goal_keeper_ids = [x.split("_")[1] for x in goal_keeper_cols]
        for id in goal_keeper_ids:
            player_col = f"Player_{id}"
            player_col_video = f"Player_{id}_video"
            goal_keeper_col = f"Goalkeeper_{id}"
            goal_keeper_col_video = f"Goalkeeper_{id}_video"
            if player_col in df.columns and player_col_video in df.columns:
                df[goal_keeper_col] = df[player_col].combine_first(df[goal_keeper_col])
                df[goal_keeper_col_video] = df[player_col_video].combine_first(df[goal_keeper_col_video])
                df.drop(columns=[player_col, player_col_video], inplace=True)

        cols = [x for x in df.columns if "Ball" not in x and "video" in x]
        TEMPORAL_THRESHOLD = int(self.fps * 1.1)

        player_video_cols = [x for x in cols if "Player" in x and "video" in x]
        goalkeeper_video_cols = [x for x in cols if "Goalkeeper" in x and "video" in x]

        to_merge = []

        # Should only merge based on video coordinates
        for col in cols:
            if "Player" in col:
                candidate_cols = player_video_cols
            elif "Goalkeeper" in col:
                candidate_cols = goalkeeper_video_cols
            else:
                print("(Should not see this): Error in column name:", col)
                continue

            last_valid_index_col = df[col].last_valid_index()
            first_valid_index_col = df[col].first_valid_index()
            for candidate in candidate_cols:
                if col == candidate:
                    continue
                first_valid_index_candidate = df[candidate].first_valid_index()
                last_valid_index_candidate = df[candidate].last_valid_index()

                # If there is an overlap, ignore
                if (
                    last_valid_index_col is not None
                    and first_valid_index_candidate is not None
                    and (last_valid_index_col >= first_valid_index_candidate or last_valid_index_candidate >= first_valid_index_col)
                ):
                    continue

                # Check which appears first
                if first_valid_index_candidate < first_valid_index_col:  # candidate appears first so we want the last
                    first_valid_index = first_valid_index_col
                    first_valid_val = df[col].loc[first_valid_index]
                    last_valid_index = last_valid_index_candidate
                    last_valid_val = df[candidate].loc[last_valid_index]
                else:
                    first_valid_index = first_valid_index_candidate
                    first_valid_val = df[candidate].loc[first_valid_index]
                    last_valid_index = last_valid_index_col
                    last_valid_val = df[col].loc[last_valid_index]

                # Essentially, we want the first valid index of the second id and the last valid index of the first appearing id
                # Condition 1 - Temporal
                if last_valid_index is None or first_valid_index is None:
                    continue
                if abs(last_valid_index - first_valid_index) > TEMPORAL_THRESHOLD:
                    continue

                # Condition 2 - Spatial
                threshold = abs(last_valid_index - first_valid_index) * 10

                dist = calculate_distance(last_valid_val, first_valid_val)
                if dist > threshold:
                    continue

                # Condition 3 - Team
                id = col.split("_")[1]
                id = int(id)
                candidate_id = candidate.split("_")[1]
                candidate_id = int(candidate_id)

                # Edge case: If we could not determine the team previously, it will not appear in the team mapping.
                # then, we just assume that this unidentified player can belong to any team
                if id in team_mapping and candidate_id in team_mapping:
                    if team_mapping[id] != team_mapping[candidate_id]:
                        continue

                # Merge
                to_merge.append((col, candidate))
            # Add the real world coordinates
        merge_real = []
        for a, b in to_merge:
            merge_real.append((a.replace("_video", ""), b.replace("_video", "")))

        to_merge.extend(merge_real)
        merged_cols = {}
        if self.debug:
            print(f"Merging {len(to_merge)} columns")
            print("To Merge:", to_merge)

        def find_root(col):
            # Find the root column to merge into
            while col in merged_cols:
                col = merged_cols[col]
            return col

        # Merge
        for col, candidate in to_merge:
            root_col = find_root(col)
            root_candidate = find_root(candidate)

            if root_col != root_candidate:
                df[root_col] = df[root_col].combine_first(df[root_candidate])
                df.drop(columns=[root_candidate], inplace=True)
                merged_cols[root_candidate] = root_col

        return df

    def parse_ball_detections_with_kalman(self, detections: list, num_to_init: int = 5, filter: bool = True, threshold: int = 100):
        init_vals = []
        non_none_init_vals = 0
        i = 0
        num_removed = 0
        while True:
            if (non_none_init_vals >= 2) and (len(init_vals) >= num_to_init):
                break
            curr = detections[i]
            if curr is not None:
                init_vals.append(curr[0])
                non_none_init_vals += 1
            else:
                init_vals.append(None)
            i += 1
            if i == len(detections):
                break

        if non_none_init_vals < 2:
            print("Not enough non-None coordinates to initialize Kalman Filter")
            return detections

        # Interpolate the initial values and backfill

        init_x = [x[0] if x is not None else None for x in init_vals]
        init_y = [x[1] if x is not None else None for x in init_vals]
        init_x = pd.Series(init_x).interpolate(method="linear").bfill().ffill().tolist()
        init_y = pd.Series(init_y).interpolate(method="linear").bfill().ffill().tolist()
        init_vals = [(x, y) for x, y in zip(init_x, init_y)]
        velocities = [(init_vals[i][0] - init_vals[i - 1][0], init_vals[i][1] - init_vals[i - 1][1]) for i in range(1, len(init_vals))]
        avg_velocity = (np.mean([x[0] for x in velocities]), np.mean([x[1] for x in velocities]))
        kf = KalmanFilter(initial_state=init_vals[0], initial_velocity=avg_velocity)
        ball_positions = []
        prev_pos = None
        prev_idx = None
        for i, candidates in enumerate(detections):
            if candidates is None or len(candidates) == 0:
                ball_positions.append(None)
                continue
            if len(candidates) == 1:
                measurement = np.array([[np.float32(candidates[0][0])], [np.float32(candidates[0][1])]])
            else:
                # Select the candidate closest to the prediction
                prediction = kf.predict()
                predicted_pos = (prediction[0, 0], prediction[1, 0])
                distances_from_pred = [np.linalg.norm(np.array(candidate) - np.array(predicted_pos)) for candidate in candidates]
                if prev_pos is not None:
                    distances_from_prev = [np.linalg.norm(np.array(candidate) - np.array(prev_pos)) for candidate in candidates]
                    distances = [0.5 * dist_pred + 0.5 * dist_prev for dist_pred, dist_prev in zip(distances_from_pred, distances_from_prev)]
                else:
                    distances = distances_from_pred
                best_candidate = candidates[np.argmin(distances)]
                measurement = np.array([[np.float32(best_candidate[0])], [np.float32(best_candidate[1])]])

            if filter:
                if prev_pos is not None:
                    # This is not the first detection, so we compute the distance from the previous detection
                    dist = calculate_distance((measurement[0, 0], measurement[1, 0]), prev_pos)[0]
                    # compute threshold based on the distance from 2 frames ago and the previous frame
                    # print(threshold, dist, prev_pos, two_frames_pos)
                    if dist > threshold * (i - prev_idx):
                        # If the distance is too large, we assume that the detection is incorrect
                        ball_positions.append(None)
                        num_removed += 1
                    else:
                        # If the distance is reasonable, we correct the Kalman filter
                        kf.correct(measurement)
                        prediction = kf.predict()
                        ball_positions.append((measurement[0, 0], measurement[1, 0]))
                        prev_pos = measurement
                        prev_idx = i
                else:
                    # If this is the first detection, we just correct the Kalman filter
                    kf.correct(measurement)
                    ball_positions.append((measurement[0, 0], measurement[1, 0]))
                    prev_pos = measurement
                    prev_idx = i
            else:
                ball_positions.append((measurement[0, 0], measurement[1, 0]))

        if self.debug and filter:
            print(f"Removed {num_removed} detections")
        return ball_positions

    def get_team_mapping(self):  # This is pretty slow
        counts = {}
        # First pass: Get the frequency of colors detected for each player
        for frame, coord in zip(self.frames, self.coords):
            coords_for_frame = self.coords[coord]
            coordinates_dict = coords_for_frame.get("Coordinates", {})
            if "Player" not in coordinates_dict or len(coordinates_dict["Player"]) == 0:
                continue
            curr_crops = [item["BBox"] for item in coordinates_dict["Player"].values()]
            for player_id, item in coordinates_dict["Player"].items():
                player_id = int(player_id)

                bbox = item["BBox"]
                x1, y1, x2, y2 = bbox
                curr_size = (x2 - x1) * (y2 - y1)
                # determine amount of overlap with other crops
                num_overlaps = 0
                max_overlap = 0
                for crop in curr_crops:
                    if crop == bbox:
                        continue
                    x1_, y1_, x2_, y2_ = crop
                    x_overlap = max(0, min(x2, x2_) - max(x1, x1_))
                    y_overlap = max(0, min(y2, y2_) - max(y1, y1_))
                    overlap = x_overlap * y_overlap
                    max_overlap = max(max_overlap, overlap)
                    if overlap > 0:
                        num_overlaps += 1
                prop_overlap = max_overlap / curr_size
                if prop_overlap > 0.35:
                    continue
                crop = frame[y1:y2, x1:x2]
                indiv_counts = self.detect_color(crop)
                if player_id not in counts:
                    counts[player_id] = {}
                for color, count in indiv_counts:
                    if color not in counts[player_id]:
                        counts[player_id][color] = 0
                    counts[player_id][color] += 1 - prop_overlap

        out = {player_id: max(color_count, key=color_count.get) for player_id, color_count in counts.items()}

        # Second pass to fix outliers
        most_common = Counter(out.values()).most_common(2)
        id_map = {color: i for i, (color, _) in enumerate(most_common)}
        team_mapping = {}
        for player_id, color in out.items():
            if color in id_map:
                team_mapping[player_id] = id_map[color]
            else:  # This is an outlier -> not in the 2 most common colors
                # Go back to the original counts and pick the most common color out of the 2 most common colors
                color_count = counts[player_id]
                color_count = [(color, count) for color, count in color_count.items() if color in id_map]
                if len(color_count) == 0:
                    print(f"Unable to determine team for player {player_id}")
                    continue
                color_count = sorted(color_count, key=lambda x: x[1], reverse=True)

                team_mapping[player_id] = id_map[color_count[0][0]]

        return team_mapping

    def detect_color(self, image):  # Get counts based on HSV after using KMeans to segment
        # Ref: https://github.com/abdullahtarek/football_analysis/blob/main/team_assigner/team_assigner.py

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use the RGB image to cluster (RGB works better than HSV for some reason)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(rgb_image.reshape(-1, 3))
        labels = kmeans.labels_
        labels = labels.reshape(image.shape[:2])
        corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        non_player_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 if non_player_cluster == 0 else 0
        mask = labels == player_cluster
        player_mask = mask.astype(np.uint8) * 255

        # use top half
        # player_mask = player_mask[: player_mask.shape[0] // 2, :]
        # hsv_image = hsv_image[: hsv_image.shape[0] // 2, :]
        hsv_image = cv2.bitwise_and(hsv_image, hsv_image, mask=player_mask)
        color_count = {color: 0 for color in color_ranges.keys()}
        masks = []
        for color, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            color_mask = cv2.inRange(hsv_image, lower, upper)
            color_mask = cv2.bitwise_and(color_mask, color_mask, mask=player_mask)
            masks.append(color_mask)

            color_count[color] += cv2.countNonZero(color_mask)

        color_count["red"] += color_count.pop("red2")

        color_count = [(color, count) for color, count in color_count.items() if count > 0]
        color_count = sorted(color_count, key=lambda x: x[1], reverse=True)

        return color_count


class KalmanFilter:
    def __init__(self, initial_state, initial_velocity):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.statePre = np.array([initial_state[0], initial_state[1], initial_velocity[0], initial_velocity[1]], dtype=np.float32).reshape(-1, 1)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1e-5
        self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1e-1
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

    def predict(self):
        return self.kalman.predict()

    def correct(self, measurement):
        self.kalman.correct(measurement)

# eagle/realtime_processor.py
class RealTimeProcessor:
    def __init__(self, fps: int, team_buffer_size: int = 120):
        self.fps = fps
        self.team_buffer_size = team_buffer_size
        self.player_color_buffer = {}  # {player_id: [colors]}
        self.team_mapping = {}
        self.frame_idx = 0
        self.color_ranges = {
            "red": [(0, 100, 100), (10, 255, 255)], "red2": [(160, 100, 100), (179, 255, 255)],
            "blue": [(96, 100, 100), (125, 255, 255)], "white": [(0, 0, 200), (180, 30, 255)],
            "black": [(0, 0, 0), (180, 255, 50)],
        }

    def update(self, frame, frame_data):
        self._update_team_buffer(frame, frame_data)
        if len(self.team_mapping) < 2 and self.frame_idx > self.team_buffer_size:
            self.assign_teams()

        self.frame_idx += 1

    def _update_team_buffer(self, frame, frame_data):
        coordinates = frame_data.get("Coordinates", {})
        for entity_type in ["Player", "Goalkeeper"]:
            if entity_type in coordinates:
                for player_id, data in coordinates[entity_type].items():
                    x1, y1, x2, y2 = data["BBox"]
                    crop = frame[y1:y2, x1:x2]
                    dominant_color = self._get_dominant_color(crop)

                    if player_id not in self.player_color_buffer:
                        self.player_color_buffer[player_id] = deque(maxlen=self.team_buffer_size)
                    self.player_color_buffer[player_id].append(dominant_color)

    def _get_dominant_color(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        max_count = 0
        dominant_color = None
        for color, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            count = cv2.countNonZero(mask)
            if count > max_count:
                max_count = count
                dominant_color = color
        return dominant_color

    def assign_teams(self):
        player_primary_colors = {}
        for player_id, colors in self.player_color_buffer.items():
            if colors:
                primary_color = Counter(colors).most_common(1)[0][0]
                player_primary_colors[player_id] = primary_color

        if not player_primary_colors:
            return

        # Simple clustering: find the two most common primary colors
        all_primary_colors = list(player_primary_colors.values())
        if len(all_primary_colors) < 2: return

        top_colors = [color for color, count in Counter(all_primary_colors).most_common(2)]
        if len(top_colors) < 2: return

        team_color_map = {top_colors[0]: 0, top_colors[1]: 1}

        for player_id, color in player_primary_colors.items():
            if color in team_color_map:
                self.team_mapping[player_id] = team_color_map[color]

    def get_team_mapping(self):
        return self.team_mapping