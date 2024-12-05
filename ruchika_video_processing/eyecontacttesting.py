import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def eyecontact_check():
    cap = cv2.VideoCapture('hemt5.mp4')  # Replace with 0 for webcam
    eye_contact_scores = []
    frame_count = 0
    score_count=0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w, _ = frame.shape
        print(frame.shape)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with Mediapipe Face Mesh
        results = face_mesh.process(rgb_frame)

        # Eye landmark indices
        left_eye_indices = [33, 133, 160, 159, 158, 157, 173, 153, 144, 145, 153, 154]  # Left eye
        right_eye_indices = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373]  # Right eye
        nose_tip = 1  # Nose tip landmark
        left_eye_corner = 33
        right_eye_corner = 362

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                try:
                    # Get coordinates for normalization (eye corners)
                    
                    left_corner_x = int(face_landmarks.landmark[left_eye_corner].x * w)
                    left_corner_y = int(face_landmarks.landmark[left_eye_corner].y * h)
                    right_corner_x = int(face_landmarks.landmark[right_eye_corner].x * w)
                    right_corner_y = int(face_landmarks.landmark[right_eye_corner].y * h)

                    # Calculate face width as distance between eye corners
                    face_width = calculate_distance((left_corner_x, left_corner_y), (right_corner_x, right_corner_y))
                    print(face_width, "face width")
                    # Calculate the average position of the irises
                    left_iris_x = sum([face_landmarks.landmark[i].x for i in left_eye_indices]) / len(left_eye_indices)
                    left_iris_y = sum([face_landmarks.landmark[i].y for i in left_eye_indices]) / len(left_eye_indices)
                    right_iris_x = sum([face_landmarks.landmark[i].x for i in right_eye_indices]) / len(right_eye_indices)
                    right_iris_y = sum([face_landmarks.landmark[i].y for i in right_eye_indices]) / len(right_eye_indices)

                    # Convert to pixel coordinates
                    left_iris_px = (int(left_iris_x * w), int(left_iris_y * h))
                    right_iris_px = (int(right_iris_x * w), int(right_iris_y * h))

                    # Get nose position
                    nose_x = int(face_landmarks.landmark[nose_tip].x * w)
                    nose_y = int(face_landmarks.landmark[nose_tip].y * h)

                    # Calculate the average distance from irises to nose
                    left_distance = calculate_distance(left_iris_px, (nose_x, nose_y))
                    print(left_distance,"left_d")
                    right_distance = calculate_distance(right_iris_px, (nose_x, nose_y))
                    print(right_distance,"right_d")
                    # Normalize distances by the face width
                    normalized_left_distance = left_distance / face_width
                    normalized_right_distance = right_distance / face_width

                    # Calculate the score (lower means closer to center, indicating eye contact)
                    score = (normalized_left_distance + normalized_right_distance) / 2
                    eye_contact_scores.append(score)
                    if score>1:
                        score-=1
                    score_count=score_count+score
                    # Print the score
                    print(f'Frame {frame_count}: Eye contact score = {score:.2f}')

                    # Optional visualization
                    cv2.circle(frame, left_iris_px, 3, (0, 255, 0), -1)
                    cv2.circle(frame, right_iris_px, 3, (0, 255, 0), -1)
                    cv2.circle(frame, (nose_x, nose_y), 3, (255, 0, 0), -1)

                except IndexError as e:
                    print(f'Index error: {e}')

        # Display the frame
        cv2.imshow('Eye Contact Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(score_count/frame_count)
    cap.release()
    cv2.destroyAllWindows()

# Run the eye contact check function
eyecontact_check()
