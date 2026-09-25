import numpy as np


# MediaPipe Face Mesh landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _landmark_to_point(landmark, width, height):
    """Convert a normalized MediaPipe landmark into pixel coordinates."""

    return np.array([
        landmark.x * width,
        landmark.y * height,
    ])


def calculate_ear(landmarks, eye_indices, width, height):
    """
    Calculate the Eye Aspect Ratio (EAR) for one eye.
    """

    points = [
        _landmark_to_point(
            landmarks[index],
            width,
            height,
        )
        for index in eye_indices
    ]

    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)

    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_eye_features(landmarks, width, height):
    """
    Calculate eye-related features for a detected face.

    Returns:
        Dictionary containing left EAR, right EAR,
        and average EAR.
    """

    left_ear = calculate_ear(
        landmarks,
        LEFT_EYE,
        width,
        height,
    )

    right_ear = calculate_ear(
        landmarks,
        RIGHT_EYE,
        width,
        height,
    )

    average_ear = (left_ear + right_ear) / 2.0

    return {
        "left_ear": left_ear,
        "right_ear": right_ear,
        "average_ear": average_ear,
    }