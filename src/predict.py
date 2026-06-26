"""Mandatory prediction program.

This file will ask the user for a mileage, load theta values from
model/params.json, and print the estimated car price.
"""

from utils import estimate_price, load_params


def ask_mileage() -> float:
    """Ask the user for a valid mileage."""
    try:
        mileage = float(input("Prediction of price for mileage: "))
    except ValueError:
        raise ValueError("Mileage must be a number")

    if mileage < 0:
        raise ValueError("Mileage cannot be negative")

    return mileage


def main() -> None:
    """Load parameters and print the estimated price."""
    print("=== Welcome to ft_linear_regression ===")
    try:
        input_mileage = ask_mileage()
        theta0, theta1 = load_params()
        predicted_price = estimate_price(input_mileage, theta0, theta1)
    except ValueError as error:
        print(f"Error: {error}")
        return

    print(f"Predicted price for mileage {input_mileage:.2f} is {predicted_price:.2f}")


if __name__ == "__main__":
    main()
