from collections import deque
import numpy as np
import cv2
from soccer_util_last import * 
import matplotlib.pyplot as plt

# Global state
player_history = {}  # player_id -> deque of (frame_idx, x, y, team)
player_stats = {}  # player_id -> (max_speed, distance)
off_ball_runs = {}

def update_player_history(player_id, position_2d, frame_idx, team=None):
    """Store the recent history of positions for a player."""
    if player_id not in player_history:
        player_history[player_id] = deque(maxlen=None)
    player_history[player_id].append((frame_idx, position_2d[0], position_2d[1], team))

def update_player_stats(player_id, speed, distance):
    """Update player statistics with max speed and distance."""
    if player_id not in player_stats:
        player_stats[player_id] = (speed, distance)
    else:
        max_speed, total_distance = player_stats[player_id]
        player_stats[player_id] = (max(max_speed, speed), total_distance + distance)

    return player_stats[player_id][1]


def calculate_speed_distance_mps(player_id, fps):
    """Return player speed in meters per second using the last two positions."""
    history = player_history.get(player_id)
    if history and len(history) >= 2:
        
        (f1, x1, y1, team1), (f2, x2, y2, team2) = history[-2], history[-1]

        # transfrom pixels to meters 
        (x1, y1), (x2, y2) = convert_list_pixels_to_meters([(x1, y1), (x2, y2)])
        dt = (f2 - f1) / fps
        if dt == 0:
            return 0.0, 0.0
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # in meters
        return (distance / dt) , distance # m/s
    return 0.0, 0.0

def convert_point_pixels_to_meters(x_coord, y_coord):
    """Convert pixel coordinates to meters based on the field scale."""
    scale_x, scale_y = get_scale_factors()
    x = x_coord / scale_x
    y = y_coord / scale_y
    return x, y

def convert_list_pixels_to_meters(pixels_coords):
    meters_list = []
    for coord in pixels_coords:
        x, y = convert_point_pixels_to_meters(coord[0], coord[1])
        meters_list.append((x, y))
    return meters_list

def reset_player_history():
    """Reset the player history and stats."""
    global player_history, player_stats
    player_history = {}
    player_stats = {}

def draw_player_with_speed(map_img, player_id, point, group, color, speed):
    """Draw the player on the field with their speed label."""
    x, y = int(point[0]), int(point[1])
    cv2.circle(map_img, (x, y), 14, color, -1)
    speed_text = f"{speed:.1f} px/s"
    cv2.putText(map_img, speed_text, (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def process_frame_players(player_2d_positions, player_id_list, frame_idx, fps):
    """Updates history, calculates speed, and prepares draw info."""
    draw_data = []
    for idx, player_id in enumerate(player_id_list):
        pt = player_2d_positions[idx]
        update_player_history(player_id, pt, frame_idx)
        speed = calculate_speed(player_id, fps)
        group = player_id_to_group.get(player_id)
        color = group_colors.get(group)
        draw_data.append((player_id, pt, group, color, speed))
    return draw_data


def get_player_history():
    """Return the player history."""
    return player_history
def get_player_stats():
    """Return the player statistics."""
    return player_stats
def get_top_distance_players(top_n=5):
    """Return the top N players with the highest distance covered."""
    sorted_players = sorted(player_stats.items(), key=lambda item: item[1][1], reverse=True)
    return sorted_players[:top_n]



def display_top_distance_players(top_n=5):
    """Display a bar chart of the top N players with the highest distance covered."""
    top_players = get_top_distance_players(top_n)
    player_ids = [player_id for player_id, _ in top_players]
    distances = [distance for _, (_, distance) in top_players]

    plt.figure(figsize=(10, 6))
    plt.bar(player_ids, distances, color='skyblue')
    plt.xlabel('Player ID')
    plt.ylabel('Distance Covered (m)')
    plt.title(f'Top {top_n} Players by Distance Covered')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def detect_off_ball_run(player_id, ball_owner_id, fps, min_run_speed=3.0, min_duration=20):
    """Detect if a player is making an off-ball run based on speed and lack of ball possession."""
    history = player_history.get(player_id)
    if not history or len(history) < min_duration:
        return
    
    # Check recent history for a consistent run
    recent_positions = list(history)[-min_duration:]
    speeds = []
    run_distance = 0.0

    for i in range(1, len(recent_positions)):
        (_, x1, y1, _), (_, x2, y2, _) = recent_positions[i - 1], recent_positions[i]
        (x1, y1), (x2, y2) = convert_list_pixels_to_meters([(x1, y1), (x2, y2)])
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        run_distance += distance
        speeds.append(distance * fps)

    avg_speed = np.mean(speeds)

    # Only log runs above speed threshold and not in possession
    if avg_speed > min_run_speed and player_id != ball_owner_id:
        start_position = (recent_positions[0][1], recent_positions[0][2])  # (x, y) in pixels
        end_position = (recent_positions[-1][1], recent_positions[-1][2])  # (x, y) in pixels
        if player_id not in off_ball_runs:
            off_ball_runs[player_id] = []
        off_ball_runs[player_id].append((start_position, end_position, run_distance))
        return True
    
    return False

def print_off_ball_runs(player_id):
    runs = off_ball_runs.get(player_id, [])
    for run in runs:
        start, end, dist = run
        print(f"Run from frame {start} to {end} covering {dist:.2f}m")
