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
