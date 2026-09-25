import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


FEATURES = [
    "average_ear",
    "eye_diff",
    "ear_change",
    "ear_rolling_mean",
    "ear_rolling_std",
]

DATA_FILE = "data/studyvision_features.csv"


def main():
    print("=" * 70)
    print("StudyVision - Leave-One-Session-Out Evaluation")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    sessions = sorted(df["session_id"].unique())

    print()
    print(f"Total sessions: {len(sessions)}")
    print(f"Total samples:  {len(df)}")

    print()
    print("=" * 70)
    print("Evaluating unseen sessions")
    print("=" * 70)

    results = []

    for test_session in sessions:
        train_df = df[df["session_id"] != test_session]
        test_df = df[df["session_id"] == test_session]

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            train_df[FEATURES],
            train_df["activity"],
        )

        predictions = model.predict(test_df[FEATURES])

        accuracy = accuracy_score(
            test_df["activity"],
            predictions,
        )

        results.append(
            {
                "session": test_session,
                "activity": test_df["activity"].iloc[0],
                "samples": len(test_df),
                "accuracy": accuracy,
            }
        )

        print(
            f"{test_session:<18} "
            f"activity={test_df['activity'].iloc[0]:<14} "
            f"samples={len(test_df):>5} "
            f"accuracy={accuracy:.4f}"
        )

    results_df = pd.DataFrame(results)

    print()
    print("=" * 70)
    print("Session-Level Results")
    print("=" * 70)

    print()

    for activity in sorted(results_df["activity"].unique()):
        activity_results = results_df[
            results_df["activity"] == activity
        ]

        print(
            f"{activity:<18} "
            f"sessions={len(activity_results)} "
            f"mean_accuracy={activity_results['accuracy'].mean():.4f}"
        )

    print()
    print("=" * 70)
    print("Overall Results")
    print("=" * 70)

    mean_accuracy = results_df["accuracy"].mean()

    weighted_accuracy = (
        (results_df["accuracy"] * results_df["samples"]).sum()
        / results_df["samples"].sum()
    )

    print()
    print(f"Mean session accuracy:     {mean_accuracy:.4f}")
    print(f"Sample-weighted accuracy:  {weighted_accuracy:.4f}")

    print()
    print("Best session:")
    best = results_df.loc[results_df["accuracy"].idxmax()]
    print(
        f"  {best['session']} "
        f"({best['activity']}) → {best['accuracy']:.4f}"
    )

    print()
    print("Worst session:")
    worst = results_df.loc[results_df["accuracy"].idxmin()]
    print(
        f"  {worst['session']} "
        f"({worst['activity']}) → {worst['accuracy']:.4f}"
    )

    print()
    print("=" * 70)
    print("Validation sessions")
    print("=" * 70)

    validation_results = results_df[
        results_df["session"].isin(
            ["validation_001", "validation_002"]
        )
    ]

    if len(validation_results) > 0:
        print(
            validation_results[
                ["session", "activity", "samples", "accuracy"]
            ].to_string(index=False)
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()