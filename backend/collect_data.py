import cv2
import numpy as np
import os
import mediapipe as mp

# Path for exported data, numpy arrays
DATA_PATH = 'MP_Data'

# Your sign language classes (customize as needed)
actions = np.array([
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'hello', 'thankyou', 'loveyou', 'space'
])

no_sequences = 30      # Number of videos you want to record per action
sequence_length = 30   # Frames per video

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

if __name__ == "__main__":
    # Create folders for each action and sequences
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except FileExistsError:
                pass

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):

                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to grab frame")
                        break

                    image, results = mediapipe_detection(frame, holistic)
                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)


                    if frame_num == 0:
                        cv2.putText(image, f'Starting collection for {action} - Sequence {sequence}',
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.imshow('Webcam Feed', image)
                        cv2.waitKey(2000)  # Wait 2 seconds before starting

                    else:
                        cv2.putText(image, f'{action} Seq: {sequence} Frame: {frame_num}',
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.imshow('Webcam Feed', image)

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

    cap.release()
    cv2.destroyAllWindows()
