# Week 4 – Logistics Forecasting & Optimization

This repository contains the Python programs for the Week 4 internship task.

## Program Files

- `generate_data.py` – Generates the simulated logistics dataset.
- `eda.py` – Performs exploratory data analysis.
- `model.py` – Trains, evaluates and tunes regression models.
- `optimize.py` – Performs transport-mode optimization using Linear Programming.
- `visualize.py` – Generates analysis and optimization plots.
- `logistics_data.csv` – Logistics shipment dataset.

## Models Used

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Tuned Random Forest using GridSearchCV

## Optimization

The optimization uses:

- 24-hour delivery SLA
- Transport-mode capacity constraints
- Transportation cost
- Predicted delivery time

The optimization is solved using SciPy's `linprog`.

## Requirements

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

## Run

```bash
python generate_data.py
python eda.py
python model.py
python optimize.py
python visualize.py
```
