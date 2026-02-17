import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


class MilkPredictor:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, df):
        features = [
            "milking_count",
            "milking_minutes",
            "activity_neck",
            "activity_leg"
        ]

        X = df[features]
        y = df["milk_yield"]

        self.model.fit(X, y)

    def predict(self, data):
        features = np.array([[
            data["milking_count"],
            data["milking_minutes"],
            data["activity_neck"],
            data["activity_leg"]
        ]])

        prediction = self.model.predict(features)
        return prediction[0]
