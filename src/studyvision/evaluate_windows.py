import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


DATA_FILE = "data/studyvision_features.csv"

FEATURES = [
    "average_ear",
    "eye_diff",
    "ear_change",
    "ear_rolling_mean",
    "ear_rolling_std",
]

# Number of samples in each prediction window.
# 30 samples is roughly 1 second if the camera is running around 30 FPS.
WINDOW_SIZE = 30


def create_windows(df):
    windows = []

    for session_id, session_df in df.groupby("session_id"):
        session_df = session_df.reset_index(drop=True)

        for start in range(
            0,
            len(session_df) - WINDOW_SIZE + 1,
            WINDOW_SIZE,
        ):
            window = session_df.iloc[
                start:start + WINDOW_SIZE
            ]

            # The activity label is constant within our
            # normal-study sessions.
            activity = window["activity"].iloc[0]

            features = {
                feature: window[feature].mean()
                for feature in FEATURES
            }

            features["session_id"] = session_id
            features["activity"] = activity

            windows.append(features)

    return pd.DataFrame(windows)


def main():
    print("=" * 70)
    print("StudyVision - Window-Level Evaluation")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    print()
    print("Raw samples:", len(df))
    print("Sessions:", df["session_id"].nunique())

    windows = create_windows(df)

    print()
    print("Window size:", WINDOW_SIZE, "samples")
    print("Total windows:", len(windows))

    print()
    print("=" * 70)
    print("Leave-One-Session-Out Window Evaluation")
    print("=" * 70)

    sessions = sorted(windows["session_id"].unique())

    results = []

    for test_session in sessions:

        train_df = windows[
            windows["session_id"] != test_session
        ]

        test_df = windows[
            windows["session_id"] == test_session
        ]

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            train_df[FEATURES],
            train_df["activity"],
        )

        predictions = model.predict(
            test_df[FEATURES]
        )

        accuracy = accuracy_score(
            test_df["activity"],
            predictions,
        )

        results.append(
            {
                "session": test_session,
                "activity": test_df["activity"].iloc[0],
                "windows": len(test_df),
                "accuracy": accuracy,
            }
        )

        print(
            f"{test_session:<18} "
            f"activity={test_df['activity'].iloc[0]:<14} "
            f"windows={len(test_df):>4} "
            f"accuracy={accuracy:.4f}"
        )

    results_df = pd.DataFrame(results)

    print()
    print("=" * 70)
    print("Activity-Level Results")
    print("=" * 70)

    for activity in sorted(
        results_df["activity"].unique()
    ):
        activity_results = results_df[
            results_df["activity"] == activity
        ]

        print(
            f"{activity:<18} "
            f"sessions={len(activity_results)} "
            f"mean_accuracy="
            f"{activity_results['accuracy'].mean():.4f}"
        )

    print()
    print("=" * 70)
    print("Overall Results")
    print("=" * 70)

    mean_accuracy = results_df["accuracy"].mean()

    weighted_accuracy = (
        (
            results_df["accuracy"]
            * results_df["windows"]
        ).sum()
        / results_df["windows"].sum()
    )

    print()
    print(
        f"Mean session accuracy:    "
        f"{mean_accuracy:.4f}"
    )

    print(
        f"Window-weighted accuracy: "
        f"{weighted_accuracy:.4f}"
    )

    print()
    print("=" * 70)
    print("Validation Sessions")
    print("=" * 70)

    validation = results_df[
        results_df["session"].isin(
            ["validation_001", "validation_002"]
        )
    ]

    print(
        validation.to_string(index=False)
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()