from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


DATA_FILE = Path("data/studyvision_features.csv")
MODEL_FILE = Path("models/studyvision_random_forest.joblib")

FEATURES = [
    "average_ear",
    "eye_diff",
    "ear_change",
    "ear_rolling_mean",
    "ear_rolling_std",
]

TARGET = "activity"


def main():
    print("=" * 60)
    print("StudyVision - Model Training")
    print("=" * 60)
    print()

    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print()

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print()

    print("Classification Report:")
    print(classification_report(y_test, predictions))

    print("Feature Importance:")
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=False)

    print(importance.round(4).to_string())
    print()

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
