import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Function to check for eye contact
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def eyecontact_check():
    cap = cv2.VideoCapture('video_5.mp4')  # Replace with 0 for webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with Mediapipe Face Mesh
        results = face_mesh.process(rgb_frame)

        # Landmark indices for specific parts
        left_eye_indices = [33, 133, 160, 159, 158, 157, 173, 153, 144, 145, 153, 154]  # Left eye
        right_eye_indices = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373]  # Right eye
        nose_indices = [6] # All nose landmarks

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                try:
                    # Draw the left eye, right eye, and full nose landmarks
                    for i in left_eye_indices + right_eye_indices + nose_indices:
                        x = int(face_landmarks.landmark[i].x * w)
                        y = int(face_landmarks.landmark[i].y * h)
                        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

                        # Label the landmarks
                        if i == nose_indices[0]:  # Label only one point of the nose
                            cv2.putText(frame, 'Nose', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                        elif i in left_eye_indices[:1]:  # Label only one point of the left eye
                            cv2.putText(frame, 'Left Eye', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                        elif i in right_eye_indices[:1]:  # Label only one point of the right eye
                            cv2.putText(frame, 'Right Eye', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                except Exception as e:
                    print(f"Error processing landmarks: {e}")

        # Display the frame
        cv2.imshow('Face Mesh with Eye and Nose Landmarks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the function
eyecontact_check()
