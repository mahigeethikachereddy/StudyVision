import cv2


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open the camera.")
        return

    print("Camera opened successfully.")
    print("Press 'q' to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read a frame.")
            break

        cv2.imshow("StudyVision - Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()