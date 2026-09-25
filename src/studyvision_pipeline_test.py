import cv2

from studyvision.face_mesh import FaceMesh
from studyvision.features.eye import calculate_eye_features


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return

    face_mesh = FaceMesh()

    print("StudyVision pipeline started.")
    print("Look at the camera.")
    print("Press 'q' to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read camera frame.")
                break

            height, width, _ = frame.shape

            faces = face_mesh.process(frame)

            if faces:
                landmarks = faces[0].landmark

                eye_features = calculate_eye_features(
                    landmarks,
                    width,
                    height,
                )

                average_ear = eye_features["average_ear"]

                cv2.putText(
                    frame,
                    f"Average EAR: {average_ear:.3f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    "FACE DETECTED",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
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
                "StudyVision Pipeline Test",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        face_mesh.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()