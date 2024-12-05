import mediapipe as mp
import cv2
import json

# Initialize MediaPipe solutions
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Replace 'video_5.mp4' with the path to your input video file
video_path = 'video_5.mp4'
cap = cv2.VideoCapture(video_path)

# Get the video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the new video without overlay
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize MediaPipe models
pose = mp_pose.Pose()
face_mesh = mp_face_mesh.FaceMesh()
hands = mp_hands.Hands()

# Initialize a list to store confidence data for each frame
confidence_data = []

frame_count = 0  # To keep track of frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if video ends

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process each frame with Pose, FaceMesh, and Hands
    pose_results = pose.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    # Calculate basic confidence indicators
    confidence_score = 0

    # Check posture (e.g., shoulder orientation)
    if pose_results.pose_landmarks:
        left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if abs(left_shoulder.y - right_shoulder.y) < 0.05:
            confidence_score += 1  # Upright posture

    # Check eye contact (approximate by face orientation)
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        nose_tip = face_landmarks.landmark[1]
        if 0.45 < nose_tip.x < 0.55:  # Assume eye contact if near center
            confidence_score += 1

    # Check hand gesture (e.g., open palm vs. closed fist)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            confidence_score += 1  # Example score for expressive hands

    # Calculate the timestamp for the current frame
    timestamp = frame_count / fps

    # Append the frame's confidence data to the list
    confidence_data.append({
        "frame": frame_count,
        "timestamp": timestamp,
        "confidence_score": confidence_score
    })

    # Write the frame to the output video without overlay
    out.write(frame)

    # Show video frame (optional, for real-time display)
    cv2.imshow("Video", frame)

    # Increment the frame count
    frame_count += 1

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Save confidence data to a JSON file
json_output_path = 'confidence_data.json'
with open(json_output_path, 'w') as json_file:
    json.dump(confidence_data, json_file, indent=4)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved as {output_path}")
print(f"Confidence data saved as {json_output_path}")
