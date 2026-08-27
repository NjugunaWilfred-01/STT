#!/usr/bin/env python3
import sys
# import os
import cv2
import numpy as np
from mediapipe.python.solutions import holistic as mp_holistic


def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return np.array([])
    
    keypoints_list = []

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            frame_keypoints = []

            # Pose (33 landmarks * 3 = 99 values)
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    frame_keypoints.extend([lm.x, lm.y, lm.z])
            else:
                frame_keypoints.extend([0.0] * 99)

            # Left hand (21 landmarks * 3 = 63 values)
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    frame_keypoints.extend([lm.x, lm.y, lm.z])
            else:
                frame_keypoints.extend([0.0] * 63)

            # Right hand (21 landmarks * 3 = 63 values)
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    frame_keypoints.extend([lm.x, lm.y, lm.z])
            else:
                frame_keypoints.extend([0.0] * 63)

            # Face (468 landmarks * 3 = 1404 values)
            if results.face_landmarks:
                for lm in results.face_landmarks.landmark:
                    frame_keypoints.extend([lm.x, lm.y, lm.z])
            else:
                frame_keypoints.extend([0.0] * 1404)

            keypoints_list.append(frame_keypoints)

    cap.release()
    
    if len(keypoints_list) == 0:
        return np.array([])
    
    return np.array(keypoints_list, dtype=np.float32)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_worker.py <input_video> <output_npy>")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_npy = sys.argv[2]
    
    keypoints = extract_keypoints_from_video(input_video)
    
    if keypoints.shape[0] > 0:
        np.save(output_npy, keypoints)
        print(f"OK:{output_npy}")
    else:
        print(f"FAIL:{input_video}")