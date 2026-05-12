import unittest
from datetime import datetime

from predictive_model import WaterConsumptionPredictor


class WaterConsumptionPredictorTests(unittest.TestCase):
    def test_predict_exact_feature_combination(self) -> None:
        predictor = WaterConsumptionPredictor()
        timestamps = [
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 10, 5),
        ]
        values = [100.0, 120.0, 80.0]
        predictor.fit(timestamps, values)

        prediction = predictor.predict_one(datetime(2024, 1, 1, 10, 0))
        self.assertAlmostEqual(prediction, 110.0, places=6)

    def test_fallback_to_hour_minute_level(self) -> None:
        predictor = WaterConsumptionPredictor()
        timestamps = [
            datetime(2024, 1, 1, 8, 0),
            datetime(2024, 1, 2, 8, 0),
        ]
        values = [50.0, 70.0]
        predictor.fit(timestamps, values)

        prediction = predictor.predict_one(datetime(2030, 8, 15, 8, 0))
        self.assertAlmostEqual(prediction, 60.0, places=6)


if __name__ == "__main__":
    unittest.main()
