"""Mandatory training program.

This file will read data/data.csv, run gradient descent, and save theta0 and
theta1 into model/params.json.
"""

from utils import estimate_price, load_dataset, save_params

LEARNING_RATE = 0.1
ITERATIONS = 10000


def normalize(value: float, min_value: float, max_value: float) -> float:
    """Scale a value into the 0 to 1 range."""
    if max_value == min_value:
        raise ValueError("Cannot normalize when min_value equals max_value")
    return (value - min_value) / (max_value - min_value)


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

    # Gradient descent
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


def main() -> None:
    """Train the model and save the learned parameters."""
    mileages, prices = load_dataset()
    theta0, theta1 = train_model(mileages, prices, LEARNING_RATE, ITERATIONS)
    save_params(theta0, theta1)

    print("Training complete")
    print(f"Rows: {len(mileages)}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Iterations: {ITERATIONS}")
    print(f"theta0: {theta0}")
    print(f"theta1: {theta1}")


if __name__ == "__main__":
    main()
