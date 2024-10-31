import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
import time
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import socket
import threading
import joblib

# Assuming this function is defined elsewhere and returns "Start" when the start gesture is detected

class HandDirectionRecognition:
    def __init__(self, classify_model_path='models/best_mlp_model.pkl'):

        # Initialize MediaPipe Hands model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7,
                                         min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.BaseOptions = python.BaseOptions(model_asset_buffer=open('models/google/gesture_recognizer.task', "rb").read())
        self.GestureRecognizer = mp.tasks.vision.GestureRecognizer
        self.GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.options = vision.GestureRecognizerOptions(base_options=self.BaseOptions,
                                                       running_mode=self.VisionRunningMode.IMAGE)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

        self.class_list = ['Start', 'Stop', 'Hold', 'Release']

        self.cap = cv2.VideoCapture(0)

        # Set the desired resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 896)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 504)
        # Get frame dimensions
        ret, frame = self.cap.read()
        self.frame_height, self.frame_width, _ = frame.shape

        # Define regions
        self.center_region = {
            'x_min': int(self.frame_width * 0.25),
            'x_max': int(self.frame_width * 0.75),
            'y_min': int(self.frame_height * 0.25),
            'y_max': int(self.frame_height * 0.75),
        }

        # Initialize variables
        self.hand_detected = False
        self.start_position = None
        self.queue_maxlen = 3
        self.max_skip_frames = 2
        self.direction_queue = deque(maxlen=self.queue_maxlen)
        self.current_direction = "Center"
        self.current_gripper = "Hold"
        self.prev_frame_time = 0

        # Initialization sequence variables
        self.state = "WAITING_FOR_START"
        self.start_signal_frames = 0
        self.countdown_start_time = 0
        self.required_start_frames = 10

        self.stop_signal_frames = 0
        self.countdown_stop_time = 0
        self.required_stop_frames = 10

        self.neutral_signal_frames = 0
        self.countdown_neutral_time = 0
        self.required_neutral_frames = 10

        # Z-coordinate tracking for forward/backward detection
        self.start_z = None
        self.z_threshold_forward = 0.013  # Adjust this value to fine-tune forward/backward sensitivity
        # self.z_threshold_backward = 0.0065
        self.z_threshold_backward = 0.025
        self.start_z_list = np.array([])
        self.z_list = np.array([])
        self.skip_frame = 0

        # Initialize socket and threading
        self.is_running = True
        self.host = "192.168.111.139"
        self.port = 2209

        self.model_recognizer = joblib.load(classify_model_path)
        self.xgb_label_encoder = joblib.load('models/xgb_label_encoder.pkl')

    def GestureInference(self, landmarks: np.ndarray) -> str:
        '''
            Predict the gesture label from the landmarks
            Return the predicted label as a string or None if no gesture is detected
        '''
        if len(landmarks) > 0:
            predicted_labels = self.model_recognizer.predict(landmarks)
            confidence_scores = max(self.model_recognizer.predict_proba(landmarks)[0])
            if confidence_scores >= 0.8:
                return predicted_labels
            else:
                return None
        else:
            return None


    def run(self):

        while self.cap.isOpened():
            self.skip_frame += 1
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            # Handling fps calculation
            self.new_frame_time = time.time()
            fps = round(1 / (self.new_frame_time - self.prev_frame_time))
            cv2.putText(frame, f'FPS: {str(fps)}', (80, self.frame_height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
            self.prev_frame_time = self.new_frame_time

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            hand_landmarks = []
            landmarks = []
            if results.multi_hand_landmarks:
                self.hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark]).flatten().reshape(1, -1)
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                gesture_result = self.GestureInference(landmarks)

                cv2.putText(frame, f"Gesture: {gesture_result}",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    
            


           

            cv2.imshow('3D Hand Direction Recognition', frame)

            if cv2.waitKey(5) & 0xFF == 27:  # Exit when ESC is pressed
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognizer = HandDirectionRecognition()
    recognizer.run()