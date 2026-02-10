from sklearn.model_selection import train_test_split
from IPython.display import display, Markdown
from deprecation import deprecated
from urllib.parse import unquote
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
import seaborn as sns
import os
from matplotlib.lines import Line2D

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU for TensorFlow

# region
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import pickle
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Reshape,
    Multiply,
    Dropout,
)
import keras_tuner as kt
from sklearn.cluster import KMeans

from tensorflow import keras
import json
import cv2
import os
# endregion


PATH = "/data/ai_club/SoccerStats2024/key_points"
# PATH = "./data"  # for Jonny local
LABELS_JSON = "Labels2ndDataSetVisited.json"
TARGET_SIZE = (1280,720)
# TARGET_SIZE = (710 // 5, 400 // 5)
RADIUS = 5

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
viets_field_img = os.path.join(_REPO_ROOT, "assets", "viets field.PNG")

WIDTH_FIELD = 63.74
LENGTH_FIELD = 107.79
RADIUS_MIDFIELD = 9.15
HALF_FIELD_X = LENGTH_FIELD/2
HALF_FIELD_Y = WIDTH_FIELD/2
DISTANCE_D_ENDLINE = 20.15
LENGTH_BOX = 16.5
HEIGHT_D_INTERSECTION_FROM_PENALTY = 7.31249
WIDTH_BOX = 40.3

# region DATA


def extract_file_name(image_path):
    # Remove the "?d=" prefix if present
    if "?d=" in image_path:
        image_path = image_path.split("?d=")[-1]

    # Decode URL-encoded characters (e.g., `%20` -> space)
    path = unquote(image_path)

    # Replace backslashes with forward slashes
    path = path.replace("\\", "/")

    # Extract just the filename
    file_name = os.path.basename(path)
    return file_name


def get_sample_result_information(sample_result):
    width = sample_result["original_width"]
    height = sample_result["original_height"]
    x_pct = sample_result["value"]["x"]
    y_pct = sample_result["value"]["y"]
    width_pct = sample_result["value"]["width"]
    # x_val, y_val, width_val = (
    #     int(x_pct / 100 * width),
    #     int(y_pct / 100 * height),
    #     int(width / 100 * width_pct),
    # )
    label = sample_result["value"]["keypointlabels"][0]

    return {
        "label": label,
        "x": x_pct / 100,
        "y": y_pct / 100,
        "width": width_pct / 100,
    }


def get_sample_information(sample, i):
    sample_file_name = extract_file_name(sample["data"]["img"])
    results = [
        get_sample_result_information(result)
        for result in sample["annotations"][0]["result"]
    ]

    return [
        {
            "file": sample_file_name,
            "label_frame_index": i,
            "width": result["width"],
            "x": result["x"],
            "y": result["y"],
            "label": result["label"].strip(),
        }
        for result in results
    ]


def fill_missing_labels(group, all_labels):
    existing_labels = group["label"].tolist()
    missing_labels = [
        label.strip() for label in all_labels if label not in existing_labels
    ]

    # Create rows for missing labels with placeholders
    placeholders = {
        "file": group["file"].iloc[0],
        "label_frame_index": group["label_frame_index"].iloc[0],
        "width": -1,
        "x": 0,
        "y": 0,
        "label": missing_labels,
    }
    missing_rows = pd.DataFrame(placeholders)
    return pd.concat([group, missing_rows], ignore_index=True)


def get_dataset():
    labels_data = json.load(open(f"{PATH}/{LABELS_JSON}", "r"))

    for sample in labels_data:
        assert len(sample["annotations"]) == 1

    keypoint_labels = []

    for i, sample in enumerate(labels_data):
        keypoint_labels += get_sample_information(sample, i)

    keypoint_labels_df = pd.DataFrame(keypoint_labels)

    
    
    all_labels = keypoint_labels_df.label.value_counts().keys().tolist()

    # Apply function to each frame
    df_filled = keypoint_labels_df.groupby("file", group_keys=False).apply(
        lambda group: fill_missing_labels(group, all_labels)
    )

    # Sort to maintain consistent label order within each frame
    df_filled = df_filled.sort_values(by=["label_frame_index", "label"]).reset_index(
        drop=True
    )

    return df_filled


def prepare_labels(keypoints):
    # Initialize visibility and coordinate arrays
    visibility = np.zeros(len(keypoints))  # Assume all points are not visible initially
    coordinates = np.zeros((len(keypoints), 2))  # Initialize coordinates as zeros

    # Process each keypoint
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:  # Visible point
            visibility[i] = 1  # Mark as visible
            coordinates[i] = [x, y]  # Store the (x, y) coordinates

    return visibility


def load_and_preprocess_image_and_coords_lines(
    image_or_path, coords=[0, 0], target_size=TARGET_SIZE, enhance_lines=True, resize=True
):
    # Load image with OpenCV
    
    if isinstance(image_or_path, str):
        image = cv2.imread(image_or_path)
        if image is None:
            raise ValueError(f"Image could not be loaded from path: {image_or_path}")
    elif isinstance(image_or_path, np.ndarray):
        image = image_or_path
    else:
        raise TypeError("Input must be a file path (str) or an image (numpy.ndarray)")

    initial_shape = image.shape
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for white color in HSV (tuned for better detection)
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 50, 255])

    # Create a binary mask of white pixels
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Apply morphological operations to thicken the lines
    kernel = np.ones((5, 5), np.uint8)  # Larger kernel for thicker lines
    thickened_mask = cv2.dilate(
        mask, kernel, iterations=2
    )  # More iterations = thicker lines

    # Apply mask to the original image
    result = image.copy()
    if enhance_lines:
        result[thickened_mask > 0] = [255, 255, 255]  # Set detected areas to white

    # Convert OpenCV image (BGR) to PIL image (RGB) for TensorFlow processing
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # Resize using TensorFlow's preprocessing tools
    img = tf.keras.preprocessing.image.array_to_img(result_rgb)
    if resize:
        img = img.resize(target_size)  # Resize to target dimensions
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize

    # Convert and scale coordinates
    target_width, target_height = target_size
    coords = np.array(coords, dtype=np.float32).reshape(-1, 2)

    # coords[:, 0] *= target_width / initial_shape[1]  # Scale x-coordinates
    # coords[:, 1] *= target_height / initial_shape[0]  # Scale y-coordinates

    return img_array, coords


def get_train_test_split(dataset, test_size=0.20):
    # Modify your loop to prepare the visibility and coordinates labels
    X = []
    y_visibility = []
    y_coordinates = []

    for image_file in dataset["file"].unique():
        img = f"{PATH}/images/{image_file}"
        keypoints = dataset[(dataset["file"] == image_file)][["x", "y"]].values

        # Prepare visibility and coordinates
        visibility = prepare_labels(keypoints)

        # Load and preprocess the image and keypoints
        img_resized, coordinates = load_and_preprocess_image_and_coords_lines(
            img, keypoints
        )

        # Store image and labels
        X.append(img_resized)
        y_visibility.append(visibility)
        y_coordinates.append(coordinates)

    # Convert to numpy arrays
    X = np.array(X)
    y_visibility = np.array(y_visibility)
    y_coordinates = np.array(y_coordinates)

    (
        X_train,
        X_val,
        y_visibility_train,
        y_visibility_val,
        y_coordinates_train,
        y_coordinates_val,
    ) = train_test_split(
        X, y_visibility, y_coordinates, test_size=test_size, random_state=42
    )

    return (
        X_train,
        X_val,
        y_visibility_train,
        y_visibility_val,
        y_coordinates_train,
        y_coordinates_val,
    )
# endregion
# region MODEL


@deprecated
def create_cnn_model(input_shape):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output
    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(7 * 2))  # Output layer for x and y coordinates
    model.add(layers.Reshape((7, 2)))  # Reshape to (batch_size, 7, 2) REVIEW

    return model


def create_cnn_model_classification(input_shape=(TARGET_SIZE[1], TARGET_SIZE[0], 3)):
    """
    Creates an improved CNN model with batch normalization and dropout.

    Args:
        input_shape: Tuple, the shape of the input image (height, width, channels).

    Returns:
        model: A Keras Model instance.
    """
    inputs = Input(shape=input_shape)

    # Convolutional layers
    x = Conv2D(32, (3, 3), activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)  # Added dropout

    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)  # Added dropout

    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)  # Added dropout

    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    # Coordinates output (7 keypoints, each with (x, y), linear activation)
    coordinates_output = Dense(12 * 2, activation="linear")(x)
    coordinates_output = Reshape((12, 2))(coordinates_output)

    # Visibility output (7 units, sigmoid activation)
    visibility_output = Dense(12, activation="sigmoid", name="visibility_output")(x)

    # Ensure coordinates are (0,0) when visibility is low
    coordinates_output = Multiply(name="coordinates_output")(
        [coordinates_output, visibility_output[..., None]]
    )

    # Define the model with two outputs
    model = Model(inputs=inputs, outputs=[visibility_output, coordinates_output])

    return model


class MaskedMAELoss(tf.keras.losses.Loss):
    def __init__(
        self, reduction="sum_over_batch_size", name="masked_mse_loss", **kwargs
    ):
        super().__init__(reduction=reduction, name=name, **kwargs)

    def call(self, y_true, y_pred):
        INVISIBLE = -1
        masked_coord_mse = tf.keras.losses.MAE(
            y_true[y_true[:, :, 0] != INVISIBLE], y_pred[y_true[:, :, 0] != INVISIBLE]
        )
        return masked_coord_mse


class RoundedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="rounded_accuracy", **kwargs):
        super(RoundedAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Round predictions
        y_pred_rounded = tf.round(y_pred)

        # Compute matches
        matches = tf.cast(tf.equal(y_true, y_pred_rounded), dtype=tf.float32)

        # Update counters
        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.size(y_true), dtype=tf.float32))

    def result(self):
        return self.correct / self.total

    def reset_state(self):
        self.correct.assign(0)
        self.total.assign(0)
# endregion
# region EVALUATION


def visualize_point_layout(dataset):
    for point_label in dataset.label.unique():
        label_slice = dataset[dataset.label == point_label].copy()

        # Calculate alpha and clamp it between 0 and 1
        alpha = 1 - (label_slice.label_frame_index / 125)
        alpha = alpha.clip(0.5, 1)

        plt.scatter(label_slice.x, -label_slice.y, label=point_label, alpha=alpha, s=3)

    plt.title("Label Movement Over Time")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()


def visualize_samples(X, y_coordinate, n_samples=8):
    fig, axes = plt.subplots(
        (n_samples + 1) // 4, 4, figsize=(12, 2 * (n_samples + 1) // 4)
    )
    axes = np.ravel(axes)

    indices = np.random.randint(0, X.shape[0], n_samples)

    for index, ax in zip(indices, axes):
        img = X[index].copy()
        img_height, img_width = X[index].shape[:2]

        for i in range(y_coordinate[index].shape[0]):
            x, y = y_coordinate[index, i]
            x_full, y_full = int(x * img_width), int(y * img_height)
            cv2.circle(
                img, (x_full, y_full), radius=RADIUS, color=(0, 0, 255), thickness=-1
            )

        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f"Image {index}")

    plt.tight_layout()
    plt.show()


def plot_training_run(history):

    metrics = [
        "xy_loss",
        "xy_mae",
        "visible_rounded_accuracy",
        "visible_loss",
        "loss",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = np.ravel(axes)
    axes[-1].remove()

    for metric, ax in zip(metrics, axes):
        train_metric = history.history[metric]
        val_metric = history.history[f"val_{metric}"]

        ax.plot(train_metric, label="train", color="tab:blue")
        ax.plot(val_metric, label=f"val", color="tab:orange")
        ax.set_title(metric)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric)
        ax.legend()

    plt.tight_layout()
    plt.show()


def visualize_keypoints(
    image,
    keypoints,
    labels=None,
    radius=RADIUS,
    color=(0, 0, 255),
    font_scale=1,
    font_thickness=2,
):
    # Make a copy of the image to avoid modifying the original
    image_with_keypoints = image.copy()
    kp_image_height, kp_image_width = image.shape[:2]

    # Iterate through keypoints and draw them
    for i, (x, y) in enumerate(keypoints):
        x, y = x, y  # Ensure coordinates are integers
        x_full, y_full = int(x * kp_image_width + radius), int(
            y * kp_image_height + radius
        )

        # Draw the keypoint (circle)
        cv2.circle(
            image_with_keypoints,
            (x_full, y_full),
            radius=radius,
            color=color,
            thickness=-1,
        )

        # Add label if provided
        if labels is not None:
            label = str(labels[i])
            cv2.putText(
                image_with_keypoints,
                label,
                (x_full + 10, y_full - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=(255, 255, 255),
                thickness=font_thickness,
                lineType=cv2.LINE_AA,
            )

    return image_with_keypoints


def evaluate_prediction(
    dataset, model, loss_fn, X, visibility, xy, index, show_average_points=False
):
    y_pred_vis, y_pred_xy = model.predict(X[[index], :])
    y_pred_vis, y_pred_xy = y_pred_vis[0], y_pred_xy[0]
    y_true_vis, y_true_xy = visibility[index], xy[index].copy()

    img = X[index].copy()

    loss = loss_fn(
        np.expand_dims(y_true_xy, -1),
        np.expand_dims(y_pred_xy, -1),
    )

    y_true_xy[:, 0] *= img.shape[1]
    y_true_xy[:, 1] *= img.shape[0]
    y_pred_xy[:, 0] *= img.shape[1]
    y_pred_xy[:, 1] *= img.shape[0]

    if show_average_points:
        avg_point = np.array(
            [
                xy[xy[:, i, 0] != -1, i, :].mean(axis=0).tolist()
                for i in range(xy.shape[1])
            ]
        )
        avg_point[:, 0] *= img.shape[1]
        avg_point[:, 1] *= img.shape[0]

        table = "| Point | Average x | Average y |\n|--|--|--|\n" + "\n".join(
            [
                f"|{label} | {point[0]:.3f} | {point[1]:.3f} |"
                for label, point in zip(dataset.label.unique(), avg_point)
            ]
        )

        display(Markdown(table))

    for i in range(y_pred_vis.shape[0]):
        y_pred_xy_i, y_true_xy_i = y_pred_xy[i], y_true_xy[i]
        y_pred_vis_i, y_true_vis_i = y_pred_vis[i], y_true_vis[i]

        if np.round(y_pred_vis_i) == 1:
            if y_true_vis_i == 1:
                cv2.line(
                    img, y_pred_xy_i.astype(int), y_true_xy_i.astype(int), (0, 0, 0), 1
                )

                cv2.circle(
                    img,
                    y_true_xy_i.astype(int),
                    radius=RADIUS,
                    color=(255, 0, 0),  # red
                    thickness=-1,
                )

            cv2.circle(
                img,
                y_pred_xy_i.astype(int),
                radius=RADIUS,
                color=(0, 0, 255),  # blue
                thickness=-1,
            )

        if show_average_points and np.round(y_pred_vis_i) == 1:
            cv2.circle(
                img,
                avg_point[i].astype(int),
                radius=RADIUS,
                color=(0, 255, 0),  # blue
                thickness=-1,
            )

            cv2.line(
                img, y_pred_xy_i.astype(int), avg_point[i].astype(int), (0, 0, 0), 1
            )
    

    # --- Show Image ---
    plt.imshow(np.clip(img, 0, 1))
    plt.title(f"Loss: {loss:.3f}")
    plt.axis("off")
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Prediction', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Ground Truth', markerfacecolor='red', markersize=8)
    ]

    if show_average_points:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', label='Average Point', markerfacecolor='green', markersize=8)
        )

    plt.legend(handles=legend_elements, loc='lower right', frameon=True)
    plt.show()
    
def evaluate_model_metrics(model, X, y_visibility, y_coordinates):
    # Predict outputs
    y_pred_vis, y_pred_xy = model.predict(X)

    # Initialize metric and loss
    accuracy_metric = RoundedAccuracy()
    masked_mae_loss = MaskedMAELoss()

    # Update state with predictions
    accuracy_metric.update_state(y_visibility, y_pred_vis)
    visibility_accuracy = accuracy_metric.result().numpy()

    # Compute masked MAE
    coordinate_mae = masked_mae_loss(y_coordinates, y_pred_xy).numpy()
    avg_scale = (TARGET_SIZE[0] + TARGET_SIZE[1]) / 2
    mae_pixels = coordinate_mae * avg_scale

    # Print results
    print(f"Final visibility accuracy: {visibility_accuracy:.4f}")
    print(f"Masked MAE for coordinates: {coordinate_mae:.4f}")
    print(f"Masked MAE for coordinates(pixels){mae_pixels:.4f}")

    return {
        "visibility_accuracy": visibility_accuracy,
        "masked_coordinate_mae": coordinate_mae
    }

# endregion

# region HOMOGRAPHY


dest_keypoints = {
    "Bottom Center Circle": (HALF_FIELD_X, HALF_FIELD_Y + RADIUS_MIDFIELD),
    "Bottom Midfield": (HALF_FIELD_X, WIDTH_FIELD),
    "Left Center Circle": (HALF_FIELD_X - RADIUS_MIDFIELD, HALF_FIELD_Y),
    "Left D": (DISTANCE_D_ENDLINE, HALF_FIELD_Y),
    "Left D Bottom Intersection": (LENGTH_BOX, HALF_FIELD_Y + HEIGHT_D_INTERSECTION_FROM_PENALTY),
    "Left D Top Intersection": (LENGTH_BOX, HALF_FIELD_Y - HEIGHT_D_INTERSECTION_FROM_PENALTY),
    "Right Center Circle": (HALF_FIELD_X + RADIUS_MIDFIELD, HALF_FIELD_Y),
    "Right D": (LENGTH_FIELD - DISTANCE_D_ENDLINE, HALF_FIELD_Y),
    "Right D Bottom Intersection": (LENGTH_FIELD - LENGTH_BOX, HALF_FIELD_Y + HEIGHT_D_INTERSECTION_FROM_PENALTY),
    "Right D Top Intersection": (LENGTH_FIELD - LENGTH_BOX, HALF_FIELD_Y - HEIGHT_D_INTERSECTION_FROM_PENALTY),
    "Top Center Circle": (HALF_FIELD_X, HALF_FIELD_Y - RADIUS_MIDFIELD),
    "Top Midfield": (HALF_FIELD_X, 0),

}



def get_scale_factors(field_img_path = viets_field_img):
    field_img = cv2.imread(field_img_path)
    field_img = cv2.cvtColor(field_img, cv2.COLOR_BGR2RGB)

    # Get image dimensions (in pixels)
    field_height, field_width = field_img.shape[:2]


    scale_x = field_width / LENGTH_FIELD  # Pixels per meter (width)
    scale_y = field_height / WIDTH_FIELD # Pixels per meter (height)
    
    return scale_x, scale_y


def plot_in_2D(transformed_pts):
    """Plots transformed keypoints on the field image."""
    # Load the field image
    field_img = cv2.imread(viets_field_img)

    # Convert BGR to RGB for matplotlib
    field_img = cv2.cvtColor(field_img, cv2.COLOR_BGR2RGB)


    # Plot transformed keypoints
    for i, (x, y) in enumerate(transformed_pts):
        cv2.circle(field_img, (int(x), int(y)), 5, (255, 255, 255), -1)  # FIXED

    # Display the transformed image
    plt.figure(figsize=(10, 6))
    plt.imshow(field_img)
    plt.axis("off")
    plt.title("Transformed Keypoints on Soccer Field Diagram")
    plt.show()


def calculate_H(src_pts, y_vis):
    
    # Convert to np array an scale them by image size
    src_pts = np.array([
    [x  * TARGET_SIZE[0], y  * TARGET_SIZE[1]] for x, y in src_pts
    ], dtype=np.float32)
    
    scale_x, scale_y = get_scale_factors()
    
    # Convert destination points to NumPy array and scale them
    dst_pts = np.array([
        [dest_keypoints[label][0] * scale_x, dest_keypoints[label][1] * scale_y]  
        for label in list(dest_keypoints.keys())  # Ensure matching number of points
    ], dtype=np.float32)
    
    
    src_points_filtered = src_pts[y_vis.squeeze().round().astype(int) == 1]
    dst_pts_filtered = dst_pts[y_vis.squeeze().round().astype(int) == 1]
    H, _ = cv2.findHomography(src_points_filtered, dst_pts_filtered)
    transformed_pts = cv2.perspectiveTransform(src_points_filtered.reshape(-1, 1, 2), H).reshape(-1, 2)
    return transformed_pts, H


def transform_points(src_pts, H_passed):
    src_pts = np.array([
        [x  * TARGET_SIZE[0], y  * TARGET_SIZE[1]] for x, y in src_pts
    ], dtype=np.float32)
    
    transformed_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H_passed).reshape(-1, 2)
    return transformed_pts



## TRANSFORMATIONS TESTS
# H_proj, _ = cv2.findHomography(src_pts[mask], dst_pts_all[mask], method=cv2.RANSAC, ransacReprojThreshold=3.0)
# H_proj, _ = cv2.findHomography(src_pts[mask], dst_pts_all[mask], method=cv2.LMEDS)
# H_proj, _ = cv2.findHomography(src_pts[mask], dst_pts_all[mask], method=cv2.USAC_MAGSAC)
# H_proj, _ = cv2.findHomography(src_pts[mask], dst_pts_all[mask])

# H_affine, _ = cv2.estimateAffine2D(src_pts[mask], dst_pts_all[mask])
# H_affine, _ = cv2.estimateAffinePartial2D(src_pts[mask], dst_pts_all[mask])
def compute_homography_mae_over_dataset(X, y_pred_vis, y_pred_xy, dest_keypoints, target_size, affine =False):
    """
    Evaluates homography accuracy across all frames using MAE.

    Args:
        X (np.ndarray): Input image data (frames), shape (N, H, W, 3)
        y_pred_vis (np.ndarray): Predicted visibility per frame, shape (N, num_points)
        y_pred_xy (np.ndarray): Predicted keypoint coordinates, shape (N, num_points, 2)
        dest_keypoints (dict): Dictionary of target field keypoints in meters
        target_size (tuple): (width, height) of input image (usually same as X[0].shape[:2])

    Returns:
        float: Mean Absolute Error (averaged over all frames and visible points)
    """
    total_error = 0
    total_points = 0
    num_frames = X.shape[0]

    # Convert destination keypoints to scaled pixel coordinates (static)
    scale_x, scale_y = get_scale_factors()

    dst_pts_all = np.array([
        [dest_keypoints[label][0] * scale_x, dest_keypoints[label][1] * scale_y]
        for label in dest_keypoints.keys()
    ], dtype=np.float32)

    for i in range(num_frames):
        frame_pred_xy = y_pred_xy[i]
        frame_pred_vis = y_pred_vis[i]

        # Scale predicted keypoints from normalized to pixels
        src_pts = np.copy(frame_pred_xy)
        src_pts[:, 0] *= target_size[0]
        src_pts[:, 1] *= target_size[1]

        # Visibility mask
        mask = np.round(frame_pred_vis).astype(bool)

        if np.sum(mask) >= 4:
            if not affine:
                H_proj, _ = cv2.findHomography(src_pts[mask], dst_pts_all[mask])

                if H_proj is not None:
                    projected_pts = cv2.perspectiveTransform(
                        src_pts[mask].reshape(-1, 1, 2), H_proj
                    ).reshape(-1, 2)
                     # Ground truth destination points for visible points
                    gt_pts = dst_pts_all[mask]

                    # Compute MAE for this frame
                    frame_error = np.abs(projected_pts - gt_pts)
                    total_error += np.sum(frame_error)
                    total_points += len(projected_pts) *2   # (x, y)
            else:
                # --- Affine Transformation ---
                H_affine, _ = cv2.estimateAffine2D(src_pts[mask], dst_pts_all[mask])
                if H_affine is not None:
                    projected_pts = cv2.transform(np.array([src_pts[mask]]), H_affine)[0]
                    
                    # Ground truth destination points for visible points
                    gt_pts = dst_pts_all[mask]

                    # Compute MAE for this frame
                    frame_error = np.abs(projected_pts - gt_pts)
                    total_error += np.sum(frame_error)
                    total_points += len(projected_pts) * 2  # (x, y)
    plot_in_2D(projected_pts)

    if total_points == 0:
        print("Warning: No valid frames with >= 4 visible points.")
        return None
    mean_abs_error_pixels = total_error / total_points

    mean_abs_error_meters = mean_abs_error_pixels / ((scale_x + scale_y) / 2)
    return {
        'mae_pixels': mean_abs_error_pixels,
        'mae_meters': mean_abs_error_meters
    }


def compute_projection_error_over_dataset(
    X, y_pred_vis, y_pred_xy, y_gt_vis, y_gt_xy, image_size=TARGET_SIZE, num_samples=2500
):

    scale_x, scale_y = get_scale_factors()
    total_error = 0
    total_points = 0

    for i in range(X.shape[0]):
        # Get predictions and GT for frame i
        pred_xy = y_pred_xy[i]
        pred_vis = y_pred_vis[i]
        gt_xy = y_gt_xy[i]
        gt_vis = y_gt_vis[i]
        mask = np.round(gt_vis).astype(bool)
        mask_pred = np.round(pred_vis).astype(bool)
        if np.sum(mask) < 4:
            print(i, "Has less than 4 keypoints")
            continue
        # Estimate homographies (you already have this function)
        _, H_pred = calculate_H(pred_xy, pred_vis)
        _, H_gt = calculate_H(gt_xy, gt_vis)

        if H_pred is None or H_gt is None:
            continue

        sampled_img_pts = np.random.uniform(
            [0, 0], image_size, size=(num_samples, 2)
        ).astype(np.float32)
        
        # Project to field using both homographies
        proj_pred = transform_points(pred_xy[mask_pred], H_pred)
        proj_gt = transform_points(gt_xy[mask], H_gt)
        # Distance in meters (field coords are in meters)
        if len(proj_pred) != len(proj_gt):
            print("Different visib predictions")
            continue
        frame_error = np.sum(np.linalg.norm(proj_pred - proj_gt, axis=1))
        total_error += frame_error
        total_points += len(proj_gt)  # (x, y)
    if total_points == 0:
        print("No valid frames with homography computed.")
        return None
    
    projection_error_pixels = total_error / total_points
    avg_scale = (scale_x + scale_y) / 2  # pixels per meter
    projection_error_meters = projection_error_pixels / avg_scale

    return {
        'projection_error_pixels': projection_error_pixels,
        'projection_error_meters': projection_error_meters
    }
# endregion

# region
### CLUSTERING

group_colors = {
    0: (255,255, 255),    # Red team
    1: (255, 0, 0),    # Blue team
    # Add more if needed
}
# endregion
