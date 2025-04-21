import cv2
import numpy as np
import cv2
import numpy as np

from soccer_util_last import *
from grouping_teams import extract_normalized_bottom_centers

 

def plot_2D_map(frames, bboxes, player_id_to_group, ball_positions, output_path,
                fps=30, model=None):
    
    map_image = cv2.imread(viets_field_img)
    # Get map dimensions
    map_height, map_width = map_image.shape[:2]

    # Define colors for teams and ball
    team_colors = {
        0: (255, 255, 255),  
        1: (255, 0, 0),  
        2: (0, 255, 255)   
    }
    ball_color = (0, 165, 255)  # Orange for the ball

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (map_width, map_height))

    # Initialize variables for smoothing and tracking
    H_smooth = None
    alpha = 0.8  # Smoothing factor for homography matrix
    last_group_positions = {}  # Dictionary to store last known positions of players
    last_ball_position = None  # Store the last known ball position

    for frame_idx, frame in enumerate(frames) :
        # Create a copy of the map image for drawing

        map_copy = map_image.copy()

        # Process every second frame
        if frame_idx % 2 == 0:
            # Get bounding boxes for players in the current frame
            player_boxes_frame = bboxes.get(frame_idx, {})  # {player_id: [(x1, y1, x2, y2)]}

            # Flatten bounding boxes and track player IDs
            flat_boxes = []
            player_id_list = []
            for player_id, boxes in player_boxes_frame.items():
                for box in boxes:
                    flat_boxes.append(box)
                    player_id_list.append(player_id)

            # Extract bottom-center points of bounding boxes
            player_positions = extract_normalized_bottom_centers(flat_boxes, frame.shape[:2])

            # Preprocess the frame for keypoint detection
            frame_preprocessed, coord = load_and_preprocess_image_and_coords_lines(frame)
            frame_input = np.expand_dims(frame_preprocessed, axis=0)

            # Predict visibility and keypoints using the model
            y_pred_vis, y_pred_xy = model.predict(frame_input)
            y_pred_vis = y_pred_vis.squeeze().round().astype(int)

            # If sufficient keypoints are visible, calculate homography
            if sum(y_pred_vis) >= 4:
                _, H = calculate_H(y_pred_xy.squeeze(), y_pred_vis.squeeze())
                H_smooth = alpha * H_smooth + (1 - alpha) * H if H_smooth is not None else H

                # Transform player positions to 2D field coordinates
                player_2d_positions = transform_points(player_positions, H_smooth)

                # Update last known positions of players
                last_group_positions = {}
                for idx, player_id in enumerate(player_id_list):
                    pt = player_2d_positions[idx]
                    if player_id not in last_group_positions:
                        last_group_positions[player_id] = []
                    last_group_positions[player_id].append(pt)

                # Transform the ball's position to 2D field coordinates
                if frame_idx < len(ball_positions):
                    ball_box = ball_positions[frame_idx]
                    if len(ball_box) == 4:
                        ball_position_normalized = extract_normalized_bottom_centers([ball_box], frame.shape[:2])
                        ball_2d_position = transform_points(ball_position_normalized, H_smooth)
                        last_ball_position = ball_2d_position[0]

                # Draw players on the map using their group colors
                for player_id, points in last_group_positions.items():
                    group = player_id_to_group.get(player_id)  # Get group ID for the player
                    color = team_colors.get(group, (255, 255, 255))  # Default to white if group not found
                    for pt in points:
                        x, y = int(pt[0]), int(pt[1])
                        cv2.circle(map_copy, (x, y), 14, color, -1)

        # If not processing the current frame, use the last known positions
        elif last_group_positions:
            for player_id, points in last_group_positions.items():
                group = player_id_to_group.get(player_id)
                color = team_colors.get(group, (255, 255, 255))
                for pt in points:
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(map_copy, (x, y), 14, color, -1)

        # Draw the ball's 2D position if available
        if last_ball_position is not None:
            ball_x, ball_y = int(last_ball_position[0]), int(last_ball_position[1])
            cv2.circle(map_copy, (ball_x, ball_y), 10, ball_color, -1)


        # Write the processed map frame to the output video
        out.write(map_copy)

    # Release the video writer
    out.release()
    print(f"Video saved to {output_path}")


def plot_video_3D(frames, bboxes, player_id_to_group, ball_positions, output_path="output_video.mp4", fps=30):
    # Get frame dimensions
    frame_height, frame_width = frames[0].shape[:2]
    
    # Define colors for teams and ball
    team_colors = {
        0: (255, 255, 255),  # Team 0 - Blue
        1: (255, 0, 0),  # Team 1 - Green
        2: (0, 255, 255)   # Team 2 - Yellow
    }
    ball_color = (0, 165, 255)  # Orange for the ball
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    for frame_idx, frame in enumerate(frames):
        # Copy the frame to draw on
        frame_copy = frame.copy()
        
        # Get bounding boxes for the current frame
        if frame_idx < len(bboxes):
            frame_bboxes = bboxes[frame_idx]
            
            for player_id, boxes in frame_bboxes.items():
                # Get the team of the player
                team_id = player_id_to_group.get(player_id, -1)
                color = team_colors.get(team_id, (255, 255, 255))  # Default to white if team not found
                
                for box in boxes:
                    if len(box) == 4:
                        x1, y1, x2, y2 = map(int, box)
                        # Draw the bounding box
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                        # Add player ID text
                        cv2.putText(frame_copy, f"ID: {player_id}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw the ball's bounding box if available
        if frame_idx < len(ball_positions):
            ball_box = ball_positions[frame_idx]
            ball_box = tuple(ball_box)
            if len(ball_box) == 4 :
                x1, y1, x2, y2 = map(int, ball_box)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), ball_color, 2)
                cv2.putText(frame_copy, "Ball", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ball_color, 2)
        
        # Write the frame to the video
        out.write(frame_copy)
    
    # Release the video writer
    out.release()
    print(f"Video saved to {output_path}")

