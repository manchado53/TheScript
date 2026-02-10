import numpy as np
import cv2

group_colors = {
    0: (255,255, 255),    # Red team
    1: (255, 0, 0),    # Blue team
    # Add more if needed
}

player_id_to_group = {
}


def add_player_to_group(player_id, group):
    """Add a player to a group."""
    player_id_to_group[player_id] = group

def get_group_for_player(player_id):
    """Get the group for a player."""
    return player_id_to_group.get(player_id, None)

def extract_box_colors(results, img, chest_center_ratio=0.3, patch_ratio=0.4):
    """Extract average color from the center patch of each detected box."""
    box_infos = []
    colors = []

    for box in results:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        box_width = x2 - x1
        box_height = y2 - y1

        # Calculate square patch size based on height
        patch_size = int(box_height * patch_ratio)

        # Center of the patch (x, y)
        center_x = (x1 + x2) // 2
        center_y = int(y1 + chest_center_ratio * box_height)

        # Calculate patch bounds
        patch_x1 = max(x1, center_x - patch_size // 2)
        patch_x2 = min(x2, center_x + patch_size // 2)
        patch_y1 = max(y1, center_y - patch_size // 2)
        patch_y2 = min(y2, center_y + patch_size // 2)

        patch = img[patch_y1:patch_y2, patch_x1:patch_x2]

        if patch.size > 0:
            avg_color = patch.mean(axis=(0, 1))  # LAB
        else:
            avg_color = np.array([0, 128, 128])  # Fallback: neutral gray


        box_infos.append(((x1, y1, x2, y2), avg_color))
        colors.append(avg_color)
    return box_infos, np.array(colors)

def extract_normalized_bottom_centers(boxes_xyxy, image_shape):
    """
    boxes_xyxy : (N,4) float32  [x1,y1,x2,y2] in pixel coords
    Returns     : (N,2) float32 bottom-centre points in [0,1] range
    """
    im_h, im_w = image_shape
    boxes = boxes_xyxy.astype(np.float32)
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5 / im_w
    cy =  boxes[:, 3]                / im_h
    return np.column_stack((cx, cy))