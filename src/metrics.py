"""Bonus precision program.

This file compares predictions with real prices and prints model precision
metrics.
"""

from math import sqrt
from utils import estimate_price, load_dataset, load_params


def validate_price_lists(real_prices: list[float], predicted_prices: list[float]) -> None:
    """Validate that metric input lists can be compared."""
    if len(real_prices) != len(predicted_prices):
        raise ValueError("Real and predicted price lists must have the same length")
    if not real_prices:
        raise ValueError("Price list is empty")


def mean_absolute_error(real_prices: list[float], predicted_prices: list[float]) -> float:
    """Calculate the average absolute prediction error."""
    validate_price_lists(real_prices, predicted_prices)

    total_error = 0.0
    for real_price, predicted_price in zip(real_prices, predicted_prices):
        total_error += abs(predicted_price - real_price)
    return total_error / len(real_prices)


def mean_squared_error(real_prices: list[float], predicted_prices: list[float]) -> float:
    """Calculate the average squared prediction error."""
    validate_price_lists(real_prices, predicted_prices)

    total_error = 0.0
    for real_price, predicted_price in zip(real_prices, predicted_prices):
        total_error += (predicted_price - real_price) ** 2
    return total_error / len(real_prices)


def root_mean_squared_error(
    real_prices: list[float],
    predicted_prices: list[float],
) -> float:
    """Calculate the square root of the average squared prediction error."""
    return sqrt(mean_squared_error(real_prices, predicted_prices))


def r2_score(real_prices: list[float], predicted_prices: list[float]) -> float:
    """Calculate how much variance is explained by the model."""
    validate_price_lists(real_prices, predicted_prices)

    average_price = sum(real_prices) / len(real_prices)
    total_variance = 0.0
    unexplained_variance = 0.0

    for real_price, predicted_price in zip(real_prices, predicted_prices):
        total_variance += (real_price - average_price) ** 2
        unexplained_variance += (real_price - predicted_price) ** 2

    if total_variance == 0:
        raise ValueError("Cannot calculate R2 when all real prices are equal")

    return 1 - (unexplained_variance / total_variance)


def mean_absolute_percentage_error(
    real_prices: list[float],
    predicted_prices: list[float],
) -> float:
    """Calculate the average prediction error as a percentage."""
    validate_price_lists(real_prices, predicted_prices)

    total_error = 0.0
    for real_price, predicted_price in zip(real_prices, predicted_prices):
        if real_price == 0:
            raise ValueError("Cannot calculate percentage error with zero price")
        total_error += abs((real_price - predicted_price) / real_price)
    return total_error * 100 / len(real_prices)


def main() -> None:
    """Load the trained model and print precision metrics."""
    mileages, real_prices = load_dataset()
    theta0, theta1 = load_params()
    predicted_prices = [estimate_price(mileage, theta0, theta1) for mileage in mileages]

    mae = mean_absolute_error(real_prices, predicted_prices)
    mse = mean_squared_error(real_prices, predicted_prices)
    rmse = root_mean_squared_error(real_prices, predicted_prices)
    r2 = r2_score(real_prices, predicted_prices)
    mape = mean_absolute_percentage_error(real_prices, predicted_prices)

    print("Model precision")
    print(f"Rows: {len(real_prices)}")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")
    print(f"MAPE: Average precision: {100 - mape:.2f}%")
    print(f"MAPE: Average error: {mape:.2f}%")


if __name__ == "__main__":
    main()
