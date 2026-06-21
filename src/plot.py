"""Bonus plotting program.

This file draws the dataset points and the trained regression line on the same
graph.
"""

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    raise SystemExit("matplotlib is required for plotting. Run: make install")

from utils import estimate_price, load_dataset, load_params


def plot_regression() -> None:
    """Display the dataset and the current regression line."""
    mileages, prices = load_dataset()
    theta0, theta1 = load_params()

    min_mileage = min(mileages)
    max_mileage = max(mileages)
    line_x = [min_mileage, max_mileage]
    line_y = [estimate_price(x, theta0, theta1) for x in line_x]

    plt.scatter(mileages, prices, color="tab:blue", label="Dataset")
    plt.plot(line_x, line_y, color="tab:red", label="Regression line")

    plt.title("Car price by mileage")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run the plotting program."""
    plot_regression()


if __name__ == "__main__":
    main()
