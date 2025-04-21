import cv2
import numpy as np
from soccer_util_last import *  # Assuming this contains the necessary utility functions
from distance_speed_stimator import get_player_history

def create_player_heatmap(player_id):
    """
    Create a heatmap for a player.
    
    - resolution: pixels per meter (e.g., 5 → heatmap will be 525x340)
    - field_size: (length, width) in meters
    - field_image_path: optional image to overlay heatmap on
    """
    image_field = cv2.imread(viets_field_img)

    height_px, width_px = image_field.shape[:2]
    heatmap = np.zeros((height_px, width_px), dtype=np.float32)

    player_history = get_player_history()
    print(f"Player history: {player_history}")
    # Accumulate player positions
    for idx, x_px, y_px, group in player_history[player_id]:
        print(f"Frame {idx} position: ({x_px}, {y_px})")
        if 0 <= x_px < width_px and 0 <= y_px < height_px:
            heatmap[int(y_px), int(x_px)] += 1

    # Smooth it
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=10, sigmaY=10)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Optional: overlay on a field image
    # field_img = cv2.imread(viets_field_img)
    # field_img = cv2.resize(field_img, (width_px, height_px))
    blended = cv2.addWeighted(image_field, 0.5, heatmap_color, 0.5, 0)

    return blended

