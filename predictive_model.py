#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Tuple


FeatureKey = Tuple[int, ...]


def parse_timestamp(value: str) -> datetime:
    raw = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(raw)


def build_features(ts: datetime) -> Dict[str, FeatureKey]:
    return {
        "full": (ts.year, ts.month, ts.day, ts.weekday(), ts.hour, ts.minute),
        "month_day_weekday_hour": (ts.month, ts.day, ts.weekday(), ts.hour, ts.minute),
        "weekday_hour": (ts.weekday(), ts.hour, ts.minute),
        "hour": (ts.hour, ts.minute),
    }


class WaterConsumptionPredictor:
    def __init__(self) -> None:
        self._means: Dict[str, Dict[FeatureKey, float]] = {
            "full": {},
            "month_day_weekday_hour": {},
            "weekday_hour": {},
            "hour": {},
        }
        self._global_mean: float = 0.0

    def fit(self, timestamps: Iterable[datetime], values: Iterable[float]) -> None:
        grouped: Dict[str, Dict[FeatureKey, List[float]]] = {
            "full": defaultdict(list),
            "month_day_weekday_hour": defaultdict(list),
            "weekday_hour": defaultdict(list),
            "hour": defaultdict(list),
        }
        all_values: List[float] = []

        for ts, value in zip(timestamps, values):
            numeric_value = float(value)
            all_values.append(numeric_value)
            features = build_features(ts)
            for level, key in features.items():
                grouped[level][key].append(numeric_value)

        if not all_values:
            raise ValueError("Aucune donnée d'entraînement fournie.")

        self._global_mean = sum(all_values) / len(all_values)
        for level, level_groups in grouped.items():
            self._means[level] = {
                key: (sum(level_values) / len(level_values))
                for key, level_values in level_groups.items()
            }

    def predict_one(self, ts: datetime) -> float:
        features = build_features(ts)
        for level in ("full", "month_day_weekday_hour", "weekday_hour", "hour"):
            key = features[level]
            if key in self._means[level]:
                return self._means[level][key]
        return self._global_mean


def load_training_data(path: str, timestamp_col: str, value_col: str) -> Tuple[List[datetime], List[float]]:
    timestamps: List[datetime] = []
    values: List[float] = []

    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("Le CSV est vide.")
        missing = [col for col in (timestamp_col, value_col) if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Colonnes manquantes: {', '.join(missing)}")

        for row in reader:
            timestamps.append(parse_timestamp(row[timestamp_col]))
            values.append(float(row[value_col]))

    return timestamps, values


def time_split_mae(
    predictor: WaterConsumptionPredictor, timestamps: List[datetime], values: List[float], ratio: float = 0.2
) -> float:
    if len(values) < 10:
        return 0.0

    split_index = int(len(values) * (1 - ratio))
    train_ts = timestamps[:split_index]
    train_values = values[:split_index]
    test_ts = timestamps[split_index:]
    test_values = values[split_index:]
    predictor.fit(train_ts, train_values)

    abs_errors = [abs(predictor.predict_one(ts) - actual) for ts, actual in zip(test_ts, test_values)]
    return sum(abs_errors) / len(abs_errors) if abs_errors else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Modèle prédictif simple des consommations d'eau (pas de 5 minutes).")
    parser.add_argument("--input", required=True, help="Chemin du CSV d'entrée")
    parser.add_argument("--timestamp-col", default="timestamp", help="Nom de colonne date/heure")
    parser.add_argument("--value-col", default="consumption", help="Nom de colonne consommation")
    parser.add_argument("--predict", help="Date/heure ISO à prédire, exemple: 2026-05-12T14:35:00")
    args = parser.parse_args()

    timestamps, values = load_training_data(args.input, args.timestamp_col, args.value_col)
    predictor = WaterConsumptionPredictor()

    mae = time_split_mae(predictor, timestamps, values)
    predictor.fit(timestamps, values)

    print(f"Observations chargées: {len(values)}")
    print(f"MAE temporelle (20% de validation): {mae:.3f}")

    if args.predict:
        prediction_ts = parse_timestamp(args.predict)
        prediction = predictor.predict_one(prediction_ts)
        print(f"Prédiction pour {prediction_ts.isoformat(sep=' ')}: {prediction:.3f}")


if __name__ == "__main__":
    main()
