import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import pandas as pd

from studyvision.face_mesh import FaceMesh
from studyvision.features.eye import calculate_eye_features
from studyvision.features.temporal import add_temporal_features
from studyvision.session_report import SessionLogger


MODEL_FILE = "models/studyvision_random_forest.joblib"

FEATURES = [
    "average_ear",
    "eye_diff",
    "ear_change",
    "ear_rolling_mean",
    "ear_rolling_std",
]

WINDOW_SIZE = 30
PREDICTION_HISTORY_SIZE = 5
DISTRACTION_DELAY_SECONDS = 3.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_feature_window(feature_history):
    """
    Convert recent raw eye-feature measurements into
    the temporal feature format expected by the model.
    """

    df = pd.DataFrame(feature_history)

    df["session_id"] = "live_session"
    df["activity"] = "unknown"

    return add_temporal_features(df)


def main():
    print("=" * 60)
    print("StudyVision - Live Activity Prediction")
    print("=" * 60)
    print()

    model = joblib.load(MODEL_FILE)

    print("Model loaded successfully.")
    print("Mode: studying versus distraction observation")
    print()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return

    face_mesh = FaceMesh()

    feature_history = deque(maxlen=WINDOW_SIZE)
    prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)
    session_logger = SessionLogger(started_at=time.time())

    previous_left_ear = None
    previous_right_ear = None
    face_missing_since = None
    on_break = False

    print("Camera started.")
    print("Study normally.")
    print("The prediction will appear after enough frames are collected.")
    print("Press 'b' to start or resume a break.")
    print("Press 'q' to stop the session.")
    print()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read camera frame.")
                break

            if on_break:
                session_logger.record(time.time(), state="on_break")
                cv2.putText(
                    frame,
                    "Status: Break",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 165, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Press B to resume or Q to finish",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(
                    "StudyVision - Live Activity Prediction",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF
                if key == ord("b"):
                    on_break = False
                    face_missing_since = None
                    print("Break ended. Study session resumed.")
                elif key == ord("q"):
                    break
                continue

            height, width, _ = frame.shape

            faces = face_mesh.process(frame)

            if faces:
                face_missing_since = None
                landmarks = faces[0].landmark

                features = calculate_eye_features(
                    landmarks,
                    width,
                    height,
                )

                left_ear = features["left_ear"]
                right_ear = features["right_ear"]
                average_ear = features["average_ear"]

                eye_diff = abs(left_ear - right_ear)

                ear_change = 0.0

                if previous_left_ear is not None:
                    previous_average = (
                        previous_left_ear + previous_right_ear
                    ) / 2

                    ear_change = abs(
                        average_ear - previous_average
                    )

                previous_left_ear = left_ear
                previous_right_ear = right_ear

                feature_history.append(
                    {
                        "timestamp": time.time(),
                        "session_id": "live_session",
                        "activity": "unknown",
                        "left_ear": left_ear,
                        "right_ear": right_ear,
                        "average_ear": average_ear,
                    }
                )

                cv2.putText(
                    frame,
                    f"EAR: {average_ear:.3f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                session_logger.record(
                    time.time(),
                    state="studying",
                )

                if len(feature_history) >= WINDOW_SIZE:
                    history_df = create_feature_window(
                        list(feature_history)
                    )

                    latest = history_df.iloc[-1]

                    prediction_input = pd.DataFrame(
                        [
                            [
                                latest["average_ear"],
                                latest["eye_diff"],
                                latest["ear_change"],
                                latest["ear_rolling_mean"],
                                latest["ear_rolling_std"],
                            ]
                        ],
                        columns=FEATURES,
                    )

                    probabilities = model.predict_proba(
                        prediction_input
                    )[0]

                    prediction_index = probabilities.argmax()

                    current_prediction = model.classes_[
                        prediction_index
                    ]

                    current_confidence = probabilities[
                        prediction_index
                    ]

                    prediction_history.append(
                        current_prediction
                    )

                    counts = Counter(prediction_history)

                    smoothed_prediction = counts.most_common(1)[0][0]

                    smoothed_indices = [
                        i
                        for i, class_name in enumerate(model.classes_)
                        if class_name == smoothed_prediction
                    ]

                    smoothed_confidence = probabilities[
                        smoothed_indices[0]
                    ]

                    cv2.putText(
                        frame,
                        "Status: Studying",
                        (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        "Face detected",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        f"Window: {len(feature_history)}/{WINDOW_SIZE}",
                        (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                else:
                    cv2.putText(
                        frame,
                        "Status: Studying",
                        (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    cv2.putText(
                        frame,
                        "Preparing observation...",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            else:
                now = time.time()
                if face_missing_since is None:
                    face_missing_since = now

                missing_duration = now - face_missing_since
                distracted = missing_duration >= DISTRACTION_DELAY_SECONDS

                if distracted:
                    session_logger.record(now, state="distracted")
                cv2.putText(
                    frame,
                    "Status: Distracted" if distracted else "Face not visible",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

                if not distracted:
                    cv2.putText(
                        frame,
                        "Distraction is recorded after 3 seconds",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

            cv2.imshow(
                "StudyVision - Live Activity Prediction",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("b"):
                on_break = True
                face_missing_since = None
                session_logger.record(time.time(), state="on_break")
                print("Break started. Press 'b' to resume or 'q' to finish.")
            elif key == ord("q"):
                break

    finally:
        face_mesh.close()
        camera.release()
        cv2.destroyAllWindows()

    csv_path, report_path = session_logger.save(PROJECT_ROOT)

    print()
    print("=" * 60)
    print("StudyVision - Prediction Session Complete")
    print("=" * 60)
    print(f"Session observations saved to: {csv_path}")
    print(f"Dashboard saved to: {report_path}")


if __name__ == "__main__":
    main()
