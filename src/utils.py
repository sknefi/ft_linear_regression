"""Shared project helpers

This file contains reusable code for CSV loading and model parameter storage
"""

import csv
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "data.csv"
PARAMS_PATH = ROOT_DIR / "model" / "params.json"


def load_dataset(path: Path = DATA_PATH) -> tuple[list[float], list[float]]:
    """Load mileage and price columns from a CSV file."""
    mileages: list[float] = []
    prices: list[float] = []

    with open(path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != ["km", "price"]:
            raise ValueError("CSV header must be exactly: km,price")

        for line_number, row in enumerate(reader, start=2):
            try:
                mileage = float(row["km"])
                price = float(row["price"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric value on line {line_number}") from error

            mileages.append(mileage)
            prices.append(price)

    if not mileages:
        raise ValueError("Dataset is empty")

    return mileages, prices


def load_params(path: Path = PARAMS_PATH) -> tuple[float, float]:
    """Load theta0 and theta1 from a JSON file."""
    with open(path, encoding="utf-8") as params_file:
        params = json.load(params_file)

    try:
        theta0 = float(params["theta0"])
        theta1 = float(params["theta1"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Parameter file must contain numeric theta0 and theta1")

    return theta0, theta1


def save_params(theta0: float, theta1: float, path: Path = PARAMS_PATH) -> None:
    """Save theta0 and theta1 into a JSON file."""
    params = {
        "_description": "Saved model parameters used by src/predict.py",
        "theta0": float(theta0),
        "theta1": float(theta1),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
        f.write("\n")


def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    """Estimate price using theta0 + theta1 * mileage."""
    return theta0 + (mileage * theta1)


def normalize(value: float, min_value: float, max_value: float) -> float:
    """Scale a value into the 0 to 1 range."""
    if max_value == min_value:
        raise ValueError("Cannot normalize when min_value equals max_value")
    return (value - min_value) / (max_value - min_value)


def denormalize(value: float, min_value: float, max_value: float) -> float:
    """Convert a normalized value back to the original range."""
    return value * (max_value - min_value) + min_value


def mean_squared_error(
    mileages: list[float],
    prices: list[float],
    theta0: float,
    theta1: float,
) -> float:
    """Calculate the average squared prediction error."""
    if len(mileages) != len(prices):
        raise ValueError("Mileage and price lists must have the same length")
    if not mileages:
        raise ValueError("Dataset is empty")

    total_error = 0.0
    for mileage, price in zip(mileages, prices):
        error = estimate_price(mileage, theta0, theta1) - price
        total_error += error ** 2
    return total_error / len(mileages)


def convert_normalized_thetas(
    theta0: float,
    theta1: float,
    min_mileage: float,
    max_mileage: float,
) -> tuple[float, float]:
    """Convert normalized-mileage thetas back to real-mileage thetas."""
    mileage_range = max_mileage - min_mileage
    if mileage_range == 0:
        raise ValueError("Cannot convert thetas when all mileages are equal")

    real_theta1 = theta1 / mileage_range
    real_theta0 = theta0 - (theta1 * min_mileage / mileage_range)
    return real_theta0, real_theta1


def train_model(
    mileages: list[float],
    prices: list[float],
    learning_rate: float = 0.1,
    iterations: int = 10000,
) -> tuple[float, float]:
    """Train theta0 and theta1 with gradient descent."""
    if len(mileages) != len(prices):
        raise ValueError("Mileage and price lists must have the same length")
    if not mileages:
        raise ValueError("Dataset is empty")
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    if iterations <= 0:
        raise ValueError("Iterations must be positive")

    min_mileage = min(mileages)
    max_mileage = max(mileages)
    normalized_mileages = [normalize(mileage, min_mileage, max_mileage) for mileage in mileages]

    theta0 = 0.0
    theta1 = 0.0
    m = len(normalized_mileages)

    for _ in range(iterations):
        sum_theta0 = 0.0
        sum_theta1 = 0.0

        for mileage, price in zip(normalized_mileages, prices):
            error = estimate_price(mileage, theta0, theta1) - price
            sum_theta0 += error
            sum_theta1 += error * mileage

        tmp_theta0 = learning_rate * sum_theta0 / m
        tmp_theta1 = learning_rate * sum_theta1 / m

        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

    return convert_normalized_thetas(theta0, theta1, min_mileage, max_mileage)
