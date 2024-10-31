import cv2
import os
import numpy as np
import mediapipe as mp
import gc
import pandas as pd
import warnings
import joblib
import random
# Suppress specific warning from google.protobuf.symbol_database
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")

def crop_image(image_path: str, output_path: str):
    '''
        Crop the image to a landscape orientation and save it to the output path
    '''
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Get original dimensions
    height, width = img.shape[:2]
    new_width = 672
    new_height = 504
    # Calculate crop dimensions
    if width > height:
        # Calculate crop coordinates
        start_x = int((width - new_width) / 2)
        start_y = int((height - new_height) / 2)

        # Crop the image
        cropped_img = img[start_y:start_y + new_height, start_x:start_x + new_width]

        cv2.imwrite(output_path, cropped_img)
    else:
        print(f"Image {image_path} is not in landscape orientation. Skipping.")

def preprocess_images(input_dir:str, output_dir:str):
    '''
        Get all images in the input directory for preprocessing
    '''
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process all images in the input directory
    for sub_folder in os.listdir(input_dir):
        if not os.path.isdir(os.path.join(output_dir, sub_folder)):
            os.makedirs(os.path.join(output_dir, sub_folder))
        for image_file in os.listdir(os.path.join(input_dir, sub_folder)):
            image_path = os.path.join(input_dir, sub_folder, image_file)
            output_path = os.path.join(output_dir, sub_folder, image_file)
            if not os.path.exists(output_path) and image_path.lower().endswith(
                    ('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                crop_image(image_path, output_path)
                print(f"Processed: {image_file}")

def create_test_set(input_dir: str, output_dir: str):
    '''
        Create a test set from the input directory
    '''
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process all images in the input directory
    for sub_folder in os.listdir(input_dir):
        if not os.path.isdir(os.path.join(output_dir, sub_folder)):
            os.makedirs(os.path.join(output_dir, sub_folder))
        for image_file in os.listdir(os.path.join(input_dir, sub_folder)):
            image_path = os.path.join(input_dir, sub_folder, image_file)
            output_path = os.path.join(output_dir, sub_folder, image_file)
            if not os.path.exists(output_path) and image_path.lower().endswith(
                    ('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Error: Could not read image {image_path}")
                    continue
                height, width = img.shape[:2]

                # Get the image that is in landscape orientation
                if width < height:
                    cv2.imwrite(output_path, img)
                    print(f"Processed: {image_file}")


def handle_joints(image_path: str, output_path: str, label: str, gesture_df: pd.DataFrame, mp_hands: mp.solutions.hands, hands: mp.solutions.hands.Hands, mp_drawing: mp.solutions.drawing_utils):
    '''
        Get all coordinates of hand joints in the image, convert to a row and append to the dataframe
    '''
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and get hand landmarks
    results = hands.process(image_rgb)

    # Draw hand landmarks on the image
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) != 1:
            os.remove(image_path)
            print(f"Unexpected number of hands detected in {image_path}. Removed.")
        else:
            new_row = [image_path]
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    new_row.append(hand_landmarks.landmark[i].x)
                    new_row.append(hand_landmarks.landmark[i].y)
                    new_row.append(hand_landmarks.landmark[i].z)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Placeholder for gesture label
            new_row.append(label)  
            gesture_df.loc[len(gesture_df)] = new_row
    else:
        # Remove the image if no landmarks are found
        print(f"No hand landmarks found in {image_path}")
        os.remove(image_path)  

    # Save the output image
    cv2.imwrite(output_path, image)


def create_data_frame(dataframe_path: str):
    '''
        Create a dataframe with columns for image path, joint coordinates and gesture label
    '''
    columns = ['image_path']
    for i in range(21):
        columns.append(f'joint{i}_x')
        columns.append(f'joint{i}_y')
        columns.append(f'joint{i}_z')
    columns.append('gesture_label')
    df = pd.DataFrame(columns=columns)
    df.to_csv(dataframe_path, index=False)

def get_dataset_joints(input_dir: str, output_drawn_dir: str, dataframe_path: str):
    '''
        Get all hand joints in the images in the input directory and call the handle_joints function to process them
    '''
    # Load the mediapipe hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
    mp_drawing = mp.solutions.drawing_utils

    # Load the dataframe
    if not os.path.exists(dataframe_path):
        create_data_frame(dataframe_path)
    gesture_df = pd.read_csv(dataframe_path)

    # Create the output directory if it does not exist
    if not os.path.exists(output_drawn_dir):
        os.makedirs(output_drawn_dir)

    # Process all images in the input directory
    for sub_folder in os.listdir(input_dir):
        if not os.path.isdir(os.path.join(output_drawn_dir, sub_folder)):
            os.makedirs(os.path.join(output_drawn_dir, sub_folder))
        for image_file in os.listdir(os.path.join(input_dir, sub_folder)):
            image_path = os.path.join(input_dir, sub_folder, image_file)
            output_path = os.path.join(output_drawn_dir, sub_folder, image_file)
            label = sub_folder
            if not os.path.exists(output_path) and image_path.lower().endswith(
                    ('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                
                # Process the image with handle_joints function
                handle_joints(image_path, output_path, label, gesture_df, mp_hands, hands, mp_drawing)
                print(f"Processed: {image_file}")
    
    # Save the dataframe to a csv file
    gesture_df.to_csv(dataframe_path, index=False)


def shuffle_dataframe(dataframe_path:str):
    '''
        Shuffle the rows in the dataframe
    '''
    df = pd.read_csv(dataframe_path)
    df = df.sample(frac=1).reset_index(drop=True)
    df['gesture_label'] = df['gesture_label'].str.replace('_inverted', '')
    df.to_csv(dataframe_path, index=False)

def remove_testset(test_dir: str, removed_percent: float):
    '''
        Remove a percentage of random images from the test set for faster testing
    '''
    for sub_folder in os.listdir(test_dir):
        if not os.path.isdir(os.path.join(test_dir, sub_folder)):
            continue

        for image_file in os.listdir(os.path.join(test_dir, sub_folder)):
            # Remove the image with a probability of removed_percent
            if random.random() < removed_percent:
                os.remove(os.path.join(test_dir, sub_folder, image_file))
        print(f"Removed: {sub_folder}")

def remove_nohand_images(input_dir: str):
    '''
        Remove images that do not contain any hand landmarks
    '''
    # Load the mediapipe hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

    # Process all images in the input directory
    for sub_folder in os.listdir(input_dir):
        if not os.path.isdir(os.path.join(input_dir, sub_folder)):
            continue
        for image_file in os.listdir(os.path.join(input_dir, sub_folder)):
            image_path = os.path.join(input_dir, sub_folder, image_file)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            if results.multi_hand_landmarks:
                if len(results.multi_hand_landmarks) != 1:
                    os.remove(image_path)
                    print(f"Unexpected number of hands detected in {image_path}. Removed.")
            else:
                os.remove(image_path)
                print(f"No hand landmarks found in {image_path}. Removed.")

def draw_hand_on_image(image_path:str):
    '''
        Draw hand landmarks on the image
    '''
    # Load the mediapipe hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=3)
    mp_drawing = mp.solutions.drawing_utils

    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Hand Landmarks', image)
        cv2.waitKey(0)
    else:
        print("No hand landmarks found in the image.")

if __name__ == '__main__':
    raw_data_dir = 'hagrid_dataset_512'
    processed_dir = 'pre_processed_v1'
    drawn_dataset_path = 'drawn_dataset'
    data_frame_path = 'gestures_df.csv'

    # preprocess_images(raw_data_dir, processed_dir) # Uncomment to run
    # get_dataset_joints(processed_dir, drawn_dataset_path, data_frame_path)  # Uncomment to run
    # shuffle_dataframe(data_frame_path)
    # create_test_set('hagrid_dataset_512', 'test_set_v1')  # Uncomment to run
    # remove_testset('test_set_v1', 0.4)  # Uncomment to run
    # remove_nohand_images('test_set_v1')  # Uncomment to run
    # draw_hand_on_image('test_set\dislike\c73c160b-5164-45d5-ad19-c91f5cb38275.jpg')  # Uncomment to run