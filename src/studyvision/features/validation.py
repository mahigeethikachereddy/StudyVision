import numpy as np


MAX_EAR = 0.5
MAX_EYE_DIFFERENCE = 0.25
MAX_FRAME_JUMP = 0.20


def validate_eye_features(
    left_ear,
    right_ear,
    previous_left_ear=None,
    previous_right_ear=None,
):
    """
    Validate a pair of EAR measurements.

    Returns:
        Dictionary containing:
        - valid
        - left_valid
        - right_valid
        - reason
    """

    left_valid = np.isfinite(left_ear) and left_ear <= MAX_EAR
    right_valid = np.isfinite(right_ear) and right_ear <= MAX_EAR

    if not left_valid and not right_valid:
        return {
            "valid": False,
            "left_valid": False,
            "right_valid": False,
            "reason": "both_eyes_invalid",
        }

    if not left_valid:
        return {
            "valid": False,
            "left_valid": False,
            "right_valid": right_valid,
            "reason": "left_eye_invalid",
        }

    if not right_valid:
        return {
            "valid": False,
            "left_valid": left_valid,
            "right_valid": False,
            "reason": "right_eye_invalid",
        }

    eye_difference = abs(left_ear - right_ear)

    if eye_difference > MAX_EYE_DIFFERENCE:
        return {
            "valid": False,
            "left_valid": True,
            "right_valid": True,
            "reason": "large_eye_difference",
        }

    if previous_left_ear is not None:
        left_jump = abs(left_ear - previous_left_ear)

        if left_jump > MAX_FRAME_JUMP:
            return {
                "valid": False,
                "left_valid": True,
                "right_valid": True,
                "reason": "left_eye_jump",
            }

    if previous_right_ear is not None:
        right_jump = abs(right_ear - previous_right_ear)

        if right_jump > MAX_FRAME_JUMP:
            return {
                "valid": False,
                "left_valid": True,
                "right_valid": True,
                "reason": "right_eye_jump",
            }

    return {
        "valid": True,
        "left_valid": True,
        "right_valid": True,
        "reason": "valid",
    }