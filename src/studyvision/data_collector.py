import csv
import time
from pathlib import Path

import cv2

from studyvision.face_mesh import FaceMesh
from studyvision.features.eye import calculate_eye_features
from studyvision.features.validation import validate_eye_features


OUTPUT_FILE = Path("data/eye_features.csv")


def main():
    allowed_activities = {
        "screen_study",
        "paper_study",
        "typing_study",
        "mixed_study",
    }

    activity = input(
        "Enter activity "
        "(screen_study/paper_study/typing/mixed_study): "
    ).strip()

    session_id = input("Enter session ID (example: screen_002): ").strip()

    if activity not in allowed_activities:
        print()
        print("ERROR: Invalid activity.")
        print("Allowed activities:")
        for name in sorted(allowed_activities):
            print(f"  - {name}")
        return

    if not session_id:
        print("ERROR: Session ID cannot be empty.")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return

    face_mesh = FaceMesh()

    print()
    print("StudyVision data collector started.")
    print(f"Activity: {activity}")
    print(f"Session: {session_id}")
    print("Start studying normally. You do not need to look at the camera.")
    print("Press 'q' to stop recording.")
    print()

    start_time = time.time()
    previous_left_ear = None
    previous_right_ear = None

    total_frames = 0
    face_frames = 0
    valid_frames = 0
    rejected_frames = 0

    frames_processed = 0
    valid_samples = 0
    rejected_samples = 0
    rejection_reasons = {}

    file_exists = OUTPUT_FILE.exists()

    with OUTPUT_FILE.open("a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "session_id",
                "activity",
                "left_ear",
                "right_ear",
                "average_ear",
            ])

        try:
            while True:
                success, frame = camera.read()

                total_frames += 1

                if not success:
                    print("ERROR: Could not read camera frame.")
                    break
                frames_processed += 1

                height, width, _ = frame.shape

                faces = face_mesh.process(frame)

                timestamp = time.time() - start_time

                if faces:
                    face_frames += 1
                    landmarks = faces[0].landmark

                    features = calculate_eye_features(
                        landmarks,
                        width,
                        height,
                    )
                    validation = validate_eye_features(
                        features["left_ear"],
                        features["right_ear"],
                        previous_left_ear,
                        previous_right_ear,
                    )

                    if validation["valid"]:
                        valid_frames += 1
                        valid_samples += 1

                        previous_left_ear = features["left_ear"]
                        previous_right_ear = features["right_ear"]

                        writer.writerow([
                            f"{timestamp:.3f}",
                            session_id,
                            activity,
                            f"{features['left_ear']:.5f}",
                            f"{features['right_ear']:.5f}",
                            f"{features['average_ear']:.5f}",
                        ])

                    else:
                        rejected_frames += 1
                        rejected_samples += 1

                        reason = validation["reason"]
                        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

                    cv2.putText(
                        frame,
                        f"EAR: {features['average_ear']:.3f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        activity,
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame,
                        "RECORDING",
                        (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                else:
                    cv2.putText(
                        frame,
                        "NO FACE",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow(
                    "StudyVision - Data Collector",
                    frame,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            face_mesh.close()
            camera.release()
            cv2.destroyAllWindows()

    print()
    print("=" * 50)
    print("Session statistics")
    print("=" * 50)
    print(f"Frames processed: {frames_processed}")
    print(f"Valid samples: {valid_samples}")
    print(f"Rejected samples: {rejected_samples}")

    if rejection_reasons:
        print()
        print("Rejection reasons:")

        for reason, count in rejection_reasons.items():
            print(f"  {reason}: {count}")
    else:
        print()
        print("Rejection reasons: none")

    print()
    print()
    print("=" * 60)
    print("StudyVision - Recording Summary")
    print("=" * 60)
    print(f"Total camera frames: {total_frames}")
    print(f"Frames with detected face: {face_frames}")
    print(f"Valid eye-feature frames: {valid_frames}")
    print(f"Rejected eye-feature frames: {rejected_frames}")

    if face_frames > 0:
        rejection_rate = rejected_frames / face_frames * 100
        print(f"Validation rejection rate: {rejection_rate:.2f}%")

    print(f"Data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()