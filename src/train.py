"""
Demonstration script for an industrial-style
machine learning regression workflow with cross-validation.
"""

import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def main():
    # Generate synthetic regression data
    np.random.seed(42)
    X = np.random.rand(200, 5)
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 0.1, 200)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model definition
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Cross-validation
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")

    # Train final model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print("Cross-validation R2:", cv_scores.mean())
    print("Test R2:", r2)
    print("MAE:", mae)
    print("RMSE:", rmse)


if name == "__main__":
    main()
