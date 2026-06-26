"""Mandatory training program.

This file will read data/data.csv, run gradient descent, and save theta0 and
theta1 into model/params.json.
"""

from utils import load_dataset, mean_squared_error, save_params, train_model


LEARNING_RATE = 0.1
ITERATIONS = 10000


def main() -> None:
    """Train the model and save the learned parameters."""
    mileages, prices = load_dataset()
    theta0, theta1 = train_model(mileages, prices, LEARNING_RATE, ITERATIONS)
    save_params(theta0, theta1)

    mse = mean_squared_error(mileages, prices, theta0, theta1)

    print("Training complete")
    print(f"Rows: {len(mileages)}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Iterations: {ITERATIONS}")
    print(f"theta0: {theta0}")
    print(f"theta1: {theta1}")
    print(f"MSE: {mse}")


if __name__ == "__main__":
    main()
