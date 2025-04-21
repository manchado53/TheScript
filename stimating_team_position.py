from distance_speed_stimator import *

ZONE_THRESHOLDS = {
    'defensive_third': (0, LENGTH_FIELD/3),
    'middle_third': (LENGTH_FIELD/3, (2*LENGTH_FIELD)/3),
    'attacking_third': ((2* LENGTH_FIELD)/3, LENGTH_FIELD)
}

team_zone_counts = {
    0: {'defensive_third': 0, 'middle_third': 0, 'attacking_third': 0},
    1: {'defensive_third': 0, 'middle_third': 0, 'attacking_third': 0}
}

total_team_frames = {
    0: 0,
    1: 0
}

def reset_zone_tracking():
    for team in team_zone_counts:
        for zone in team_zone_counts[team]:
            team_zone_counts[team][zone] = 0
        total_team_frames[team] = 0


def update_zone_tracking(last_group_positions):
    for player_id, (pt_2d, _, group, _, _, _) in last_group_positions.items():
        x,_ = convert_point_pixels_to_meters(pt_2d[0], 0)
        team = group
        for zone_name, (x_min, x_max) in ZONE_THRESHOLDS.items():
            if x_min <= x < x_max:
                team_zone_counts[team][zone_name] += 1
                break
        total_team_frames[team] += 1


def get_zone_occupancy_percentages():
    """
    Returns a dictionary of zone occupancy percentages per team.
    """
    percentages = {}
    for team in team_zone_counts:
        percentages[team] = {}
        for zone in team_zone_counts[team]:
            total = total_team_frames[team]
            zone_frames = team_zone_counts[team][zone]
            percentage = (zone_frames / total) * 100 if total > 0 else 0
            percentages[team][zone] = percentage
    return percentages


def draw_zone_occupancy_overlay(zone_percentages):
    """
    Draw zone occupancy intensity map on the field image.
    Each zone is split horizontally: top = team 0, bottom = team 1.
    Color intensity corresponds to occupancy percentage.
    """
    image_field = cv2.imread(viets_field_img)
    height_px, width_px = image_field.shape[:2]
    overlay = image_field.copy()

    thirds = list(ZONE_THRESHOLDS.values())
    scale_x, scale_y = get_scale_factors()

    for i, (x_min, x_max) in enumerate(thirds):
        x1 = int(scale_x * x_min)
        x2 = int(scale_x * x_max)
        zone_height = height_px // 2

        zone_name = list(ZONE_THRESHOLDS.keys())[i]

        for team in [0, 1]:
            percentage = zone_percentages[team][zone_name] / 100.0
            base_color = group_colors[team]

            # Adjust color by percentage
            color = tuple(int(c * percentage) for c in base_color)

            # Split horizontally: top = team 0, bottom = team 1
            if team == 0:
                y1, y2 = 0, zone_height
            else:
                y1, y2 = zone_height, height_px

            # Draw the zone rectangle
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=cv2.FILLED)

            # Add percentage label
            text = f"{zone_name}: {percentage*100:.1f}%"
            text_x = x1 + 10
            text_y = y1 + 30
            cv2.putText(overlay, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Blend overlay with field image
    blended = cv2.addWeighted(overlay, 0.6, image_field, 0.4, 0)
    return blended
