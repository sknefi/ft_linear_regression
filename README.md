# ft_linear_regression

Linear regression project for the 42 subject.

## Project Structure

```text
.
├── data/
│   └── data.csv
├── model/
│   └── params.json
├── src/
│   ├── predict.py
│   ├── train.py
│   ├── plot.py
│   ├── metrics.py
│   └── utils.py
├── Makefile
├── requirements.txt
└── README.md
```

## Roles

- `src/train.py`: mandatory entrypoint that trains the model and saves theta values.
- `src/predict.py`: mandatory entrypoint that asks for mileage and estimates a price.
- `src/utils.py`: shared logic for loading data, estimating prices, training, and saving parameters.
- `src/plot.py`: bonus entrypoint for data and regression-line graph.
- `src/metrics.py`: bonus entrypoint for model precision.
- `model/params.json`: saved `theta0` and `theta1` values.
- `Makefile`: shortcut commands for setup, mandatory programs, and bonus programs.

## Commands

```sh
make all
make install
make train
make predict
make plot
make metrics
```

## Rule To Respect

The project must implement the regression and gradient descent logic manually.
Do not use helpers such as `numpy.polyfit`, `sklearn.linear_model`, or any library
that trains the model for you.
