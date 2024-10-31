import cv2
import mediapipe as mp
import numpy as np
import socket
import threading
import joblib
import warnings
import time
from collections import deque, Counter
from mediapipe.framework.formats import landmark_pb2

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")


class HandDirectionRecognition:
    def __init__(self, classify_model_path='models/best_mlp_model.pkl'):

        # Initialize MediaPipe Hands model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.9, min_tracking_confidence=0.8)
        self.mp_drawing = mp.solutions.drawing_utils

        # Load the trained model
        self.model_recognizer = joblib.load(classify_model_path)
        self.xgb_label_encoder = joblib.load('models/xgb_label_encoder.pkl')

        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)

        # Set the desired resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 896)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 504)

        # Get frame dimensions
        _ , frame = self.cap.read()
        self.frame_height, self.frame_width, _ = frame.shape
        self.prev_frame_time = 0
        self.new_frame_time = 0

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
        self.current_direction = "Center"
        self.current_gripper = "Hold"

        # Initialization sequence variables
        self.state = "WAITING_FOR_START"

        self.start_signal_frames = 0
        self.countdown_start_time = 0
        self.required_start_frames = 75

        self.stop_signal_frames = 0
        self.countdown_stop_time = 0
        self.required_stop_frames = 75

        self.neutral_signal_frames = 0
        self.countdown_neutral_time = 0
        self.required_neutral_frames = 75
        self.countdown_moving_neutral = 0

        # Z-coordinate tracking for forward/backward detection
        self.start_z = None
        self.z_threshold_forward = 0.012 
        
        self.z_threshold_backward = 0.010
        self.start_z_list = np.array([])
        self.z_list = np.array([])
        self.latest_update = time.time()
        self.waiting_next_update = 2

        # Initialize socket and threading
        self.is_running = True
        self.host = "192.168.23.128"
        self.port = 2209

    def GestureInference(self, landmarks: np.ndarray) -> str:
        '''
            Predict the gesture label from the landmarks
            Return the predicted label as a string or None if no gesture is detected
        '''
        if len(landmarks) > 0:
            predicted_labels = self.model_recognizer.predict(landmarks)
            confidence_scores = max(self.model_recognizer.predict_proba(landmarks)[0])
            if confidence_scores >= 0.9:
                return predicted_labels
            else:
                return None
        else:
            return None

    def get_palm_landmarks_z(self, mp_landmarks: landmark_pb2.NormalizedLandmarkList) -> np.ndarray:
        '''
            Get the z-coordinates of the palm landmarks
            Return the z-coordinates as a numpy array
        '''
        palm_landmarks = np.array([mp_landmarks.landmark[0].z])
        for i in range(5, 20, 4):
            palm_landmarks = np.append(palm_landmarks, mp_landmarks.landmark[i].z)
        return palm_landmarks

    def line_function(self, x1, y1, x2, y2, xp, yp):
        '''
            Calculate the line function for two points and a given point
            Return the result value of the line function
        '''
        return (y2 - y1) * (xp - x1) - (x2 - x1) * (yp - y1)

    def get_hand_direction_2d(self, mp_landmarks: landmark_pb2.NormalizedLandmarkList) -> str:
        '''
            Get the direction of the hand in 2D space, focus on wrist landmark
            Return the direction as a string
        '''
        # Get the wrist landmark
        wrist_landmark = mp_landmarks.landmark[0]
        
        # Get the x and y coordinates of the wrist landmark
        x_coord = wrist_landmark.x * self.frame_width
        y_coord = wrist_landmark.y * self.frame_height

        # Calculate the line function for the line from center corners to frame corners
        # Top-left corner
        lf_top_left = self.line_function(x1=0, y1=0, x2=self.center_region['x_min'], y2=self.center_region['y_min'], xp=x_coord, yp=y_coord)
        # Top-right corner
        lf_top_right = self.line_function(x1=self.frame_width, y1=0, x2=self.center_region['x_max'], y2=self.center_region['y_min'], xp=x_coord, yp=y_coord)
        # Bottom-left corner
        lf_bottom_left = self.line_function(x1=0, y1=self.frame_height, x2=self.center_region['x_min'], y2=self.center_region['y_max'], xp=x_coord, yp=y_coord)
        # Bottom-right corner
        lf_bottom_right = self.line_function(x1=self.frame_width, y1=self.frame_height, x2=self.center_region['x_max'], y2=self.center_region['y_max'], xp=x_coord, yp=y_coord)

        # Check the wrist joint position to determine the direction
        # if (self.center_region['x_min'] <= x_coord <= self.center_region['x_max'] and self.center_region['y_min'] <= y_coord <= self.center_region['y_max']):
        #     direction = "Center"

        if (y_coord < self.center_region['y_min'] and lf_top_left > 0 and lf_top_right < 0):
            direction = "Up"
        elif (y_coord > self.center_region['y_max'] and lf_bottom_left < 0 and lf_bottom_right > 0):    
            direction = "Down"
        elif (x_coord < self.center_region['x_min'] and lf_top_left < 0 and lf_bottom_left > 0): 
            direction = "Left"
        elif (x_coord > self.center_region['x_max'] and lf_top_right > 0 and lf_bottom_right < 0):
            direction = "Right"
        else:
            direction = "Center"
        return direction
        
    def get_hand_direction_forward_backward(self, avg_z: float) -> str:
        '''
            Get the forward/backward direction of the hand based on the average z-coordinate
            Return the direction as a string
        '''
        if self.start_z is not None:
            # Check the average z-coordinate to determine the direction
            if avg_z > (self.start_z + self.z_threshold_backward):
                print(f"Backward: {avg_z} > {self.start_z}")
                direction = "Backward"
            elif avg_z < (self.start_z - self.z_threshold_forward):
                print(f"Forward: {avg_z} < {self.start_z}")
                direction = "Forward"
            else:
                direction = "Center"
        return direction

    def process_frame(self, mp_landmarks: landmark_pb2.NormalizedLandmarkList) -> str:
        '''
            Process the hand landmarks to determine the direction of the hand
            Return the direction as a string
        '''
        
        if mp_landmarks:
            self.z_list = self.get_palm_landmarks_z(mp_landmarks)
            avg_z = np.mean(self.z_list)

            # Determine left/right/up/down movement
            direction = self.get_hand_direction_2d(mp_landmarks=mp_landmarks)
            
            if direction == "Center":
                # Determine forward/backward movement
                direction = self.get_hand_direction_forward_backward(avg_z=avg_z)

            return direction
        return None

    def draw_direction_areas(self, frame: np.ndarray):
        '''
            Draw the direction areas on the frame
        '''
        h, w = frame.shape[:2]
        cv2.line(frame, (0, 0), (self.center_region['x_min'], self.center_region['y_min']), (0, 255, 0), 2)
        cv2.line(frame, (w, 0), (self.center_region['x_max'], self.center_region['y_min']), (0, 255, 0), 2)
        cv2.line(frame, (0, h), (self.center_region['x_min'], self.center_region['y_max']), (0, 255, 0), 2)
        cv2.line(frame, (w, h), (self.center_region['x_max'], self.center_region['y_max']), (0, 255, 0), 2)
        cv2.rectangle(frame,
                      (self.center_region['x_min'], self.center_region['y_min']),
                      (self.center_region['x_max'], self.center_region['y_max']),
                      (0, 255, 0), 2)

    def send_update(self, data: str):
        '''
            Send the direction order and gripper status as a string to the server
        '''
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            try:
                client_socket.connect((self.host, self.port))
                client_socket.sendall(str(data).encode('utf-8'))
                print(f"Sent data: {data} to server")

            except ConnectionRefusedError:
                print("Connection refused. Make sure the server is running.")

    def start_client(self):
        '''
            Start the client thread for sending data to the server
        '''
        threading.Thread(target=self.run, args=(self.host, self.port), daemon=True).start()

    def display_countdown(self, frame: np.ndarray, remaining_time: int, message: str):
        '''
            Display the countdown on the frame
        '''
        cv2.putText(frame, str(remaining_time),
                    (self.frame_width // 2, self.frame_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
        cv2.putText(frame, message,
                    (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def waiting_for_start_signal(self, frame: np.ndarray, landmarks: np.ndarray, hand_landmarks: landmark_pb2.NormalizedLandmarkList):
        '''
            Wait for the start signal to perform start signal countdown and enter control mode
        '''
        # Reset the start z list
        self.start_z_list = []
        cv2.putText(frame, "Use the start signal to enter control mode",
                    (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Change the state to countdown for start signal if the start signal is detected
        if self.GestureInference(landmarks) == "rock":
            self.start_z_list = np.concatenate((self.start_z_list, self.get_palm_landmarks_z(hand_landmarks)))
            self.start_signal_frames = 0
            self.state = "COUNTDOWN_FOR_START"
            self.countdown_start_time = time.time()

    def countdown_for_start_signal(self, frame: np.ndarray, landmarks: np.ndarray, hand_landmarks: landmark_pb2.NormalizedLandmarkList):
        '''
            Perform the countdown for collecting start signal frames to enter control mode
        '''
        remaining_time = 3 - int(time.time() - self.countdown_start_time)

        # Add the z-coordinates of the palm landmarks detected in frames in this duration to the start z list
        self.start_z_list = np.concatenate((self.start_z_list, self.get_palm_landmarks_z(hand_landmarks)))

        if remaining_time > 0:
            self.display_countdown(frame, remaining_time, "Hold the start signal")
            if self.GestureInference(landmarks) == "rock":
                self.start_signal_frames += 1
        else:
            # If the required number of frames are captured, calculate the average z-coordinate and change the state to control mode
            if self.start_signal_frames >= self.required_start_frames:
                self.start_z = np.mean(self.start_z_list)
                self.state = "CONTROL_MODE"
                print("Start position captured. You can now move your hand to detect directions.")
                self.start_signal_frames = 0
            else:
                self.state = "WAITING_FOR_START"

    def countdown_for_stop_signal(self, frame, landmarks):
        stop_remaining_time = 3 - int(time.time() - self.countdown_stop_time)
        if stop_remaining_time > 0:
            self.display_countdown(frame, stop_remaining_time, "Hold the STOP signal")
            if self.GestureInference(landmarks) in ["peace", "peace_inverted"]:
                self.stop_signal_frames += 1
        else:
            if self.stop_signal_frames >= self.required_stop_frames:
                self.start_z = np.array([])
                self.start_z_list = np.array([])
                self.state = "WAITING_FOR_START"
                self.stop_signal_frames = 0
            else:
                self.state = "CONTROL_MODE"
        
    def countdown_for_neutral_signal(self, frame, landmarks):
        neutral_remaining_time = 3 - int(time.time() - self.countdown_neutral_time)
        if neutral_remaining_time > 0:
            self.display_countdown(frame, neutral_remaining_time, "Captured neutral signal...")
            if self.GestureInference(landmarks) == "one":
                self.neutral_signal_frames += 1
        else:
            if self.neutral_signal_frames >= self.required_neutral_frames:
                self.current_direction = 'Neutral'
                self.current_gripper = 'Hold'
                sending_data = str(self.current_direction) + "_" + str(self.current_gripper)
                self.send_update(sending_data)
                self.state = "MOVING_TO_NEUTRAL"
                self.countdown_moving_neutral = time.time()
                self.neutral_signal_frames = 0
            else:
                self.state = "CONTROL_MODE"

    def countdown_for_moving_to_neutral(self, frame):
        moving_neutral_remaining_time = 5 - int(time.time() - self.countdown_moving_neutral)
        if moving_neutral_remaining_time > 0:
            self.display_countdown(frame, moving_neutral_remaining_time, "Moving Sawyer back to Neutral Position...")
        else:
            self.state = "CONTROL_MODE"

    def control_mode(self, frame, landmarks, hand_landmarks):
        gesture = self.GestureInference(landmarks)
        if gesture in ["peace", "peace_inverted"]:
            self.countdown_stop_time = time.time()
            self.state = "COUNTDOWN_FOR_STOP"
            cv2.putText(frame, "Captured stop signal, exiting control mode ...",
                        (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return
        elif gesture == "fist":
            self.current_gripper = "Hold"
        elif gesture == "palm":
            self.current_gripper = "Release"
        elif gesture == "one":
            self.countdown_neutral_time = time.time()
            self.state = "COUNTDOWN_FOR_NEUTRAL_SIGNAL"
            cv2.putText(frame, "Captured neutral moving signal ...",
                        (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return
        if hand_landmarks:
            self.current_direction = self.process_frame(mp_landmarks=hand_landmarks)
            print(f"Current direction: {self.current_direction}")
            if time.time() - self.latest_update > self.waiting_next_update:
                sending_data = str(self.current_direction) + "_" + str(self.current_gripper)
                print(f"Sending new direction order: {sending_data}")
                self.send_update(sending_data)
                self.latest_update = time.time()
        cv2.putText(frame, f"Direction: {self.current_direction}",
                    (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Gripper mode: {self.current_gripper}",
                        (int(self.frame_width / 2), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    def run(self):
        '''
            Run the hand direction recognition application
        '''
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                continue

            # Proprocess the frame
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Handling fps calculation
            self.new_frame_time = time.time()
            fps = round(1 / (self.new_frame_time - self.prev_frame_time))
            cv2.putText(frame, f'FPS: {str(fps)}', (80, self.frame_height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
            self.prev_frame_time = self.new_frame_time

            # Draw direction areas
            self.draw_direction_areas(frame)
            
            # Process the hand landmarks
            results = self.hands.process(rgb_frame)
            hand_landmarks = []
            landmarks = []
            if results.multi_hand_landmarks:
                self.hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark]).flatten().reshape(1, -1)
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            else:
                self.hand_detected = False
                self.current_direction = "Center"
            
            # State machine for initialization sequence
            if self.state == "WAITING_FOR_START":
                self.waiting_for_start_signal(frame, landmarks, hand_landmarks)
            elif self.state == "COUNTDOWN_FOR_START":
                self.countdown_for_start_signal(frame, landmarks, hand_landmarks)
            elif self.state == "COUNTDOWN_FOR_NEUTRAL_SIGNAL":
                self.countdown_for_neutral_signal(frame, landmarks)
            elif self.state == "MOVING_TO_NEUTRAL":
                self.countdown_for_moving_to_neutral(frame)
            elif self.state == "CONTROL_MODE":
                self.control_mode(frame, landmarks, hand_landmarks)
            elif self.state == "COUNTDOWN_FOR_STOP":
                self.countdown_for_stop_signal(frame, landmarks)

            cv2.imshow('3D Hand Direction Recognition', frame)

            if cv2.waitKey(5) & 0xFF == 27:  # Exit when ESC is pressed
                break

        self.cap.release()
        cv2.destroyAllWindows()


 
if __name__ == "__main__":
    recognizer = HandDirectionRecognition()
    recognizer.run()