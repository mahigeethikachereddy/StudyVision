import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from studyvision.features.temporal import add_temporal_features


DATA_FILE = "data/studyvision_features.csv"


def main():
    df = pd.read_csv(DATA_FILE)

    train = df[df["session_id"] != "validation_002"].copy()
    test = df[df["session_id"] == "validation_002"].copy()

    feature_sets = {
        "all_5": [
            "average_ear",
            "eye_diff",
            "ear_change",
            "ear_rolling_mean",
            "ear_rolling_std",
        ],
        "no_average_ear": [
            "eye_diff",
            "ear_change",
            "ear_rolling_mean",
            "ear_rolling_std",
        ],
        "no_eye_diff": [
            "average_ear",
            "ear_change",
            "ear_rolling_mean",
            "ear_rolling_std",
        ],
        "no_ear_change": [
            "average_ear",
            "eye_diff",
            "ear_rolling_mean",
            "ear_rolling_std",
        ],
        "no_rolling_mean": [
            "average_ear",
            "eye_diff",
            "ear_change",
            "ear_rolling_std",
        ],
        "no_rolling_std": [
            "average_ear",
            "eye_diff",
            "ear_change",
            "ear_rolling_mean",
        ],
    }

    train = add_temporal_features(train)
    test = add_temporal_features(test)

    print("=" * 70)
    print("StudyVision - Leave-One-Feature-Out Ablation")
    print("=" * 70)
    print()
    print("Training sessions:", train["session_id"].unique().tolist())
    print("Validation session:", test["session_id"].unique().tolist())
    print("Validation samples:", len(test))
    print()

    results = {}

    for name, features in feature_sets.items():
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            train[features],
            train["activity"],
        )

        predictions = model.predict(test[features])

        accuracy = accuracy_score(
            test["activity"],
            predictions,
        )

        results[name] = accuracy

        print(f"{name:20s} Accuracy: {accuracy:.4f}")

    print()
    print("=" * 70)
    print("Ablation Summary")
    print("=" * 70)

    best_name = max(results, key=results.get)

    print()
    print(f"Best feature set: {best_name}")
    print(f"Best accuracy:    {results[best_name]:.4f}")
    print()


if __name__ == "__main__":
    main()
