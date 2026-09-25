import cv2
import mediapipe as mp


class FaceMesh:
    """Detect and track facial landmarks using MediaPipe Face Mesh."""

    def __init__(
        self,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self._mp_face_mesh = mp.solutions.face_mesh

        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame):
        """
        Process a BGR OpenCV frame.

        Returns:
            A list of detected faces, where each face contains
            MediaPipe facial landmarks.
        """

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self._face_mesh.process(rgb_frame)

        if results.multi_face_landmarks is None:
            return []

        return results.multi_face_landmarks

    def close(self):
        """Release MediaPipe resources."""
        self._face_mesh.close()