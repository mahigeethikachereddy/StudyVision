from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/studyvision_features.csv")

FEATURES = [
    "average_ear",
    "eye_diff",
    "ear_change",
    "ear_rolling_mean",
    "ear_rolling_std",
]


def main():
    df = pd.read_csv(DATA_FILE)

    print("=" * 70)
    print("StudyVision - Session Feature Analysis")
    print("=" * 70)
    print()

    print("Feature means by session:")
    print()

    session_means = (
        df.groupby(["session_id", "activity"])[FEATURES]
        .mean()
        .round(4)
    )

    print(session_means.to_string())

    print()
    print("=" * 70)
    print("Feature medians by session")
    print("=" * 70)
    print()

    session_medians = (
        df.groupby(["session_id", "activity"])[FEATURES]
        .median()
        .round(4)
    )

    print(session_medians.to_string())


if __name__ == "__main__":
    main()