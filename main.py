from src.utils import load_dataset, load_params

def main() -> None:
    """Print a short summary of loaded data and parameters."""
    mileages, prices = load_dataset()
    theta0, theta1 = load_params()

    print(f"Loaded {len(mileages)} data rows")
    print(f"First row: km={mileages[0]}, price={prices[0]}")
    print(f"Current theta0={theta0}, theta1={theta1}")


if __name__ == "__main__":
    main()
