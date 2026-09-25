import pandas as pd


def add_temporal_features(df):
    """
    Add temporal eye-movement features to the dataset.

    Features:
        eye_diff:
            Absolute difference between left and right EAR.

        ear_change:
            Absolute change in average EAR from the previous sample
            within the same session.

        ear_rolling_mean:
            Rolling mean of average EAR within each session.

        ear_rolling_std:
            Rolling standard deviation of average EAR within each session.
    """

    df = df.copy()

    # Difference between the two eyes
    df["eye_diff"] = (
        df["left_ear"] - df["right_ear"]
    ).abs()

    # Frame-to-frame EAR change
    df["ear_change"] = (
        df.groupby("session_id")["average_ear"]
        .diff()
        .abs()
        .fillna(0)
    )

    # Rolling statistics within each session
    df["ear_rolling_mean"] = (
        df.groupby("session_id")["average_ear"]
        .transform(
            lambda x: x.rolling(
                window=15,
                min_periods=1,
            ).mean()
        )
    )

    df["ear_rolling_std"] = (
        df.groupby("session_id")["average_ear"]
        .transform(
            lambda x: x.rolling(
                window=15,
                min_periods=1,
            ).std()
        )
        .fillna(0)
    )

    return df
