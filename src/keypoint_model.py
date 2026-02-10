from ultralytics import YOLO

class KeypointModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        # Training logic for YOLO pose can be implemented here if needed
        pass

    def load(self, path):
        """
        Load a YOLO pose model from the given weights file (e.g., yolov8n-pose.pt).
        """
        self.model = YOLO(path)

    def predict(self, X):
        """
        Run inference using the YOLO pose model. X can be a path to an image, a list of images, or a numpy array.
        Returns the YOLO results object.
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Call load() first.")
        return self.model.predict(X)
