import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "auto-mpg.csv"
PLOT_PATH = ROOT / "report"

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Remove car name for modeling
    if "car name" in df.columns:
        df = df.drop(columns=["car name"])

    return df

def prepare_data(df):
    X = df.drop(columns=["mpg"])
    y = df["mpg"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate(y_true, y_pred, name="model"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} results:")
    print(f" MAE: {mae:.3f}")
    print(f" MSE: {mse:.3f}")
    print(f" R²:  {r2:.3f}")
    return mae, mse, r2

def save_plot(y_test, y_pred, k):
    PLOT_PATH.mkdir(exist_ok=True)
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.xlabel("True MPG")
    plt.ylabel("Predicted MPG")
    plt.title(f"True vs Predicted MPG (k={k})")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--')
    plt.savefig(PLOT_PATH / "true_vs_predicted_mpg.png", bbox_inches="tight")
    plt.close()

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Baseline
    baseline_pred = np.repeat(y_train.mean(), len(y_test))
    evaluate(y_test, baseline_pred, "Baseline (mean)")

    best_k = None
    best_mae = float("inf")

    for k in [1,3,5,7,9,11,13,15]:
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        print(f"k={k} → MAE={mae}")
        if mae < best_mae:
            best_mae = mae
            best_k = k

    print("\nBest K:", best_k)

    model = KNeighborsRegressor(n_neighbors=best_k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate(y_test, y_pred, f"kNN(k={best_k})")

    save_plot(y_test, y_pred, best_k)

if __name__ == "__main__":
    main()
