import numpy as np
from collections import defaultdict
from soccer_util_last import viets_field_img
import cv2

class PossessionTracker:
    def __init__(self, horizontal_zones=3, vertical_zones=3, possession_radius=12):
        map_image = cv2.imread(viets_field_img)
        self.field_height = map_image.shape[0]
        self.field_width = map_image.shape[1]
        self.horizontal_zones = horizontal_zones
        self.vertical_zones = vertical_zones
        self.zone_width = self.field_width // vertical_zones
        self.zone_height = self.field_height // horizontal_zones
        self.possession_radius = possession_radius

        # Initialize counters per zone: team_id → frame count
        self.zone_possession_counts = {
            (row, col): defaultdict(int)
            for row in range(horizontal_zones)
            for col in range(vertical_zones)
        }

    def get_zone_index(self, x, y):
        col = min(int(x // self.zone_width), self.vertical_zones - 1)
        row = min(int(y // self.zone_height), self.horizontal_zones - 1)
        return row, col

    def update_possession(self, ball_position, player_positions, player_id_to_group):
        if ball_position is None or not player_positions:
            return
        ball_x, ball_y = ball_position
        zone_idx = self.get_zone_index(ball_x, ball_y)

        # Find the closest player to the ball
        min_dist = float("inf")
        closest_team = None

        for player_id, point in player_positions.items():
            team = player_id_to_group.get(player_id)
            if team == 2:  # Skip team 2 (e.g., referees or untracked players)
                continue
            px, py = tuple(point)
            dist = np.hypot(ball_x - px, ball_y - py)
            if dist < min_dist:
                min_dist = dist
                closest_team = player_id_to_group.get(player_id)

        # Count frame for the team if within the radius
        if closest_team is not None and min_dist <= self.possession_radius:
            self.zone_possession_counts[zone_idx][closest_team] += 1
            print(
                f"Zone {zone_idx}: Team {closest_team} possession count: {self.zone_possession_counts[zone_idx][closest_team]}"
            )

    def get_possession_percentages(self):
        possession_percentages = {}
        for row in range(self.horizontal_zones):
            for col in range(self.vertical_zones):
                counts = self.zone_possession_counts[(row, col)]
                total = sum(counts.values())
                possession_percentages[(row, col)] = {
                    team: (100 * count / total if total > 0 else 0)
                    for team, count in counts.items()
                }
        return possession_percentages

    def get_total_possession(self):
        """
        Calculates the total possession percentage for each team across all zones.
        Returns a dictionary with team IDs as keys and their total possession percentages as values.
        """
        total_counts = defaultdict(int)

        # Sum up possession counts for each team across all zones
        for zone, counts in self.zone_possession_counts.items():
            for team, count in counts.items():
                total_counts[team] += count

        # Calculate total possession percentages
        total_frames = sum(total_counts.values())
        total_possession = {
            team: (100 * count / total_frames if total_frames > 0 else 0)
            for team, count in total_counts.items()
        }

        return total_possession

    def print_summary(self):
        """
        Prints a summary of possession percentages for each zone.
        """
        print("\nPossession by Zone:")
        for row in range(self.horizontal_zones):
            for col in range(self.vertical_zones):
                print(f"Zone ({row}, {col}):")
                total = sum(self.zone_possession_counts[(row, col)].values())
                for team, frames in self.zone_possession_counts[(row, col)].items():
                    percent = 100 * frames / total if total > 0 else 0
                    print(f"  Team {team}: {percent:.1f}%")

    def draw_zones_with_possession(self, output_path="zones_with_possession_map.jpg"):
        """
        Draws the zones on the 2D map of the field and overlays possession percentages for each team.
        """
        # Load the field image
        field_image = cv2.imread(viets_field_img)
        field_image = cv2.cvtColor(field_image, cv2.COLOR_BGR2RGB)

        # Draw vertical and horizontal lines to separate zones
        for col in range(1, self.vertical_zones):
            x = col * self.zone_width
            cv2.line(field_image, (x, 0), (x, self.field_height), (255, 0, 0), 3)  # Blue lines
        for row in range(1, self.horizontal_zones):
            y = row * self.zone_height
            cv2.line(field_image, (0, y), (self.field_width, y), (255, 0, 0), 3)  # Blue lines

        # Add zone labels and possession percentages
        for row in range(self.horizontal_zones):
            for col in range(self.vertical_zones):
                x = int((col + 0.5) * self.zone_width)
                y = int((row + 0.5) * self.zone_height)
                cv2.putText(
                    field_image,
                    f"Zone {row},{col}",
                    (x - 50, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),  # Green text
                    2,
                    cv2.LINE_AA,
                )

                # Overlay possession percentages for each team
                percentages = self.get_possession_percentages().get((row, col), {})
                y_offset = y + 40
                for team, percent in percentages.items():
                    cv2.putText(
                        field_image,
                        f"Team {team}: {percent:.1f}%",
                        (x - 100, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),  # White text
                        2,
                        cv2.LINE_AA,
                    )
                    y_offset += 30

        # Save the image with zones and possession percentages
        cv2.imwrite(output_path, cv2.cvtColor(field_image, cv2.COLOR_RGB2BGR))
        print(f"Zones map with possession saved to {output_path}")

    def draw_team_possession_heatmap(self, team_id, output_path="team_possession_heatmap.jpg"):
        """
        Draws a heatmap of possession percentages for a given team using different tones of white.
        """
        # Load the field image
        field_image = cv2.imread(viets_field_img)
        field_image = cv2.cvtColor(field_image, cv2.COLOR_BGR2RGB)

        # Draw vertical and horizontal lines to separate zones
        for col in range(1, self.vertical_zones):
            x = col * self.zone_width
            cv2.line(field_image, (x, 0), (x, self.field_height), (255, 0, 0), 3)  # Blue lines
        for row in range(1, self.horizontal_zones):
            y = row * self.zone_height
            cv2.line(field_image, (0, y), (self.field_width, y), (255, 0, 0), 3)  # Blue lines

        # Overlay possession percentages for the given team
        possession_percentages = self.get_possession_percentages()
        for row in range(self.horizontal_zones):
            for col in range(self.vertical_zones):
                percentages = possession_percentages.get((row, col), {})
                team_percent = percentages.get(team_id, 0)

                # Calculate the intensity of white based on possession percentage
                intensity = int(255 * (team_percent / 100))
                color = (intensity, intensity, intensity)  # Shades of white

                # Fill the zone with the calculated color
                top_left = (col * self.zone_width, row * self.zone_height)
                bottom_right = ((col + 1) * self.zone_width, (row + 1) * self.zone_height)
                cv2.rectangle(field_image, top_left, bottom_right, color, -1)

                # Add possession percentage text
                x = int((col + 0.5) * self.zone_width)
                y = int((row + 0.5) * self.zone_height)
                cv2.putText(
                    field_image,
                    f"{team_percent:.1f}%",
                    (x - 30, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),  # Black text for contrast
                    2,
                    cv2.LINE_AA,
                )

        # Save the heatmap image
        cv2.imwrite(output_path, cv2.cvtColor(field_image, cv2.COLOR_RGB2BGR))
        print(f"Team possession heatmap saved to {output_path}")