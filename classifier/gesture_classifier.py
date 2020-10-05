import cv2
import numpy as np
from tensorflow.keras import models


class Classifier(object):

    def __init__(self, model_path):
        self.model = models.load_model(model_path)

    def __call__(self, image):
        image = self.preprocessing(image)
        predictions = self.model(image)
        return predictions

    @staticmethod
    def preprocessing(image):
        image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_LINEAR) / 255
        image = np.expand_dims(image, axis=0)
        return image
