SYSTEM_PYTHON ?= python3
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python3
PIP ?= $(VENV)/bin/pip3

.PHONY: all help venv install train predict plot metrics clean fclean re

# Run the automatic project workflow.
all: train metrics

# Show available commands.
help:
	@echo "Available targets:"
	@echo "  make all      - train the model and calculate precision"
	@echo "  make venv     - create the local Python virtual environment"
	@echo "  make install  - install Python dependencies"
	@echo "  make train    - train the model"
	@echo "  make predict  - predict a car price from mileage"
	@echo "  make plot     - plot dataset and regression line"
	@echo "  make metrics  - calculate model precision"
	@echo "  make clean    - remove Python cache files"
	@echo "  make fclean   - clean and reset model parameters"
	@echo "  make re       - reset model, then train again"

# Create the local Python virtual environment.
venv:
	$(SYSTEM_PYTHON) -m venv $(VENV)

# Install Python dependencies.
install: venv
	$(PIP) install -r requirements.txt

# Train theta0 and theta1.
train: venv
	$(PYTHON) src/train.py

# Ask for mileage and estimate a price.
predict: venv
	$(PYTHON) src/predict.py

# Display the dataset and regression line.
plot: install
	$(PYTHON) src/plot.py

# Display precision metrics for the trained model.
metrics: venv
	$(PYTHON) src/metrics.py

# Remove Python cache files.
clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Reset generated files and saved parameters.
fclean: clean
	$(PYTHON) -c 'import json; json.dump({"_description": "Saved model parameters used by src/predict.py.", "theta0": 0.0, "theta1": 0.0}, open("model/params.json", "w"), indent=2); print()'

# Reset and train again.
re: fclean train
