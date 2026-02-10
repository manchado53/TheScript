import cv2
import numpy as np
from possesion_tracker import PossessionTracker
from grouping_teams import player_id_to_group, group_colors, extract_normalized_bottom_centers
from distance_speed_stimator import (
    update_player_history, calculate_speed_distance_mps, update_player_stats,
    detect_off_ball_run, reset_player_history
)
from soccer_util_last import (
    viets_field_img, load_and_preprocess_image_and_coords_lines,
    calculate_H, transform_points
)
from keypoint_model import KeypointModel
from constants import path_model
import pickle

def process_video(video_path, output_path_3D, model, bboxes, ball_positions):
    print("[INFO] Starting video processing...")
    tracker = PossessionTracker(3, 3)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] FPS: {fps}, Resolution: {frame_width}x{frame_height}")
    viets_field_img_obj = cv2.imread(viets_field_img)
    viets_field_img_height, viets_field_img_width = viets_field_img_obj.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out3D = cv2.VideoWriter(output_path_3D, fourcc, fps, (frame_width, frame_height))
    H_smooth = None
    alpha = 0.8
    frame_idx = 0
    last_group_positions = {}
    reset_player_history()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video read error or finished.")
            break
        if frame_idx % 50 == 0:
            print(f"[INFO] Processing frame {frame_idx}...")
        frame_orig = frame.copy()
        frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_RGB2BGR)

        if frame_idx % 5 == 0:
            print(f"[INFO] Updating homography at frame {frame_idx}...")
            frame_preprocessed, coord = load_and_preprocess_image_and_coords_lines(frame_orig)

            predictions = model.predict(frame_preprocessed)
            print(f"[INFO] Predictions: {predictions}")
            y_pred_vis = predictions[0].boxes.cls
            y_pred_xy = predictions[0].boxes.xyxy
            # y_pred_vis = y_pred_vis.squeeze().round().astype(int)
            print(f"[INFO] Predictions at frame {frame_idx}...")
            if sum(y_pred_vis) >= 4:
                _, H = calculate_H(y_pred_xy.squeeze(), y_pred_vis.squeeze())
                H_smooth = alpha * H_smooth + (1 - alpha) * H if H_smooth is not None else H
                print(f"[INFO] Homography updated at frame {frame_idx}.")
        if frame_idx % 2 == 0 and H_smooth is not None:
            print(f"[INFO] Processing player positions at frame {frame_idx}...")
            player_boxes_frame = bboxes.get(frame_idx, {})
            flat_boxes = []
            player_id_list = []
            for player_id, boxes in player_boxes_frame.items():
                for box in boxes:
                    flat_boxes.append(box)
                    player_id_list.append(player_id)
            player_positions = extract_normalized_bottom_centers(flat_boxes, frame_orig.shape[:2])

            if sum(y_pred_vis) >= 4:
                player_2d_positions = transform_points(player_positions, H_smooth)
                draw_data = []
                for idx, player_id in enumerate(player_id_list):
                    pt_2d = player_2d_positions[idx]
                    frame_pt = player_positions[idx]
                    group = player_id_to_group.get(player_id)

                    update_player_history(player_id, pt_2d, frame_idx, group)
                    speed, last_distance = calculate_speed_distance_mps(player_id, fps)
                    total_distance = update_player_stats(player_id, speed, last_distance)

                    color = group_colors.get(group)
                    draw_data.append((player_id, pt_2d, frame_pt, group, color, speed, total_distance))
                detect_off_ball_run(10, 0, fps)

                last_group_positions = {
                    player_id: (pt_2d, frame_pt, group, color, speed, total_distance)
                    for player_id, pt_2d, frame_pt, group, color, speed, total_distance in draw_data
                }
                player_positions = {
                    player_id: pt_2d
                    for player_id, pt_2d, _, _, _, _, _ in draw_data
                }
                ball_box = ball_positions[frame_idx]
                if len(ball_box) == 4:
                    ball_position_normalized = extract_normalized_bottom_centers([ball_box], frame_orig.shape[:2])
                    ball_2d_position = transform_points(ball_position_normalized, H_smooth)
                    last_ball_position = ball_2d_position[0]
                else:
                    last_ball_position = None
                tracker.update_possession(last_ball_position, player_positions, player_id_to_group)
                print(f"[INFO] Player positions processed at frame {frame_idx}.")

        # Drawing
        print(f"[INFO] Drawing overlays at frame {frame_idx}...")
        for player_id, (pt_2d, frame_pt, group, color, speed, total_distance) in last_group_positions.items():
            frame_x, frame_y = int(frame_pt[0] * frame_width), int(frame_pt[1] * frame_height)
            speed_text = f"{speed:.1f} m/s"
            total_distance_text = f"Dist: {total_distance:.1f} m"
            cv2.circle(frame_orig, (frame_x, frame_y), 8, color, -1)
            cv2.putText(frame_orig, speed_text, (frame_x + 10, frame_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_orig, total_distance_text, (frame_x + 10, frame_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if 'ball_box' in locals() and len(ball_box) == 4:
            ball_x, ball_y = int(ball_box[0] * frame_width), int(ball_box[1] * frame_height)
            cv2.circle(frame_orig, (ball_x, ball_y), 8, (0, 255, 0), -1)
            cv2.putText(frame_orig, "Ball", (ball_x + 10, ball_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        total_possession = tracker.get_total_possession()
        team_0_possession = total_possession.get(0, 0)
        team_1_possession = total_possession.get(1, 0)
        possession_text_0 = f"Team 0: {team_0_possession:.1f}%"
        possession_text_1 = f"Team 1: {team_1_possession:.1f}%"
        box_width, box_height = 600, 70
        box_x, box_y = frame_width - box_width - 20, 20
        cv2.rectangle(frame_orig, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)
        cv2.putText(frame_orig, possession_text_0, (box_x + 10, box_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        box_y_below = box_y + box_height + 10
        cv2.rectangle(frame_orig, (box_x, box_y_below), (box_x + box_width, box_y_below + box_height), (0, 0, 0), -1)
        cv2.putText(frame_orig, possession_text_1, (box_x + 10, box_y_below + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        out3D.write(cv2.cvtColor(frame_orig, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Frame {frame_idx} written to output video.")
        frame_idx += 1

    cap.release()
    out3D.release()
    print("[INFO] Processing complete.")

if __name__ == "__main__":
    # Define or import model, bboxes, and ball_positions here
    video_path = '/data/ai_club/SoccerStats2024/30SecondsAurora.mp4'
    output_path_3D = '/data/ai_club/SoccerStats2024/AuroraVideo3D.mp4'
    model = KeypointModel()
    model.load(path_model)
    with open("/data/ai_club/SoccerStats2024/adrian_video.pkl", 'rb') as f:
        bboxes = pickle.load(f)
    with open("/data/ai_club/SoccerStats2024/ball_220frames_filled_smooth.pkl", 'rb') as f:
        ball_positions = pickle.load(f)
    process_video(video_path, output_path_3D, model, bboxes, ball_positions)
