#!/usr/bin/env python3.10
import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
import collections
from collections import deque
import io
import operator
import tflite_runtime.interpreter as tflite
import time
from edgetpu.utils import image_processing
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

Category = collections.namedtuple('Category', ['id', 'score'])

def load_cpu_interpreter(model_path):
    return tflite.Interpreter(model_path=model_path)

def load_edgetpu_interpreter(model_path):
    edgetpu_delegate = tflite.load_delegate('libedgetpu.so.1',options={"device":"usb:0","device":"usb:1"})
    return tflite.Interpreter(model_path=model_path, experimental_delegates=[edgetpu_delegate])

def load_gpu_interpreter(model_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = 'gpu'
    return tflite.interpreter(model_path=model_path)
    
def get_output(interpreter, top_k, score_threshold):
    """Returns no more than top_k categories with score >= score_threshold."""
    scores = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    categories = [
        Category(i, scores[0][i])
        for i in np.argpartition(scores, -top_k)[-top_k:]
        if scores[0][i] >= score_threshold
    ]
    return sorted(categories, key=operator.itemgetter(1), reverse=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Real-time image classification with Coral Edge TPU', add_help=False)
    parser.add_argument('--model_path', type=str, required=True, help='Path to .tflite model')
    parser.add_argument('--label_path', type=str, required=True, help='Path to label file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument('--width', type=int, default=640, help='Width of video frame')
    parser.add_argument('--height', type=int, default=480, help='Height of video frame')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'tpu', 'gpu'], help='Device to use')
    parser.add_argument('--video_device', type=int, default=0, help='Index of video capture device')
    parser.add_argument('--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    return parser.parse_args()

def load_interpreter(model_path, device):
    if device == 'cpu':
         return tflite.Interpreter(model_path=model_path)
    elif device == 'tpu':
         edgetpu_delegate = tflite.load_delegate('libedgetpu.so.1')
         return tflite.Interpreter(model_path=model_path, experimental_delegates=[edgetpu_delegate])
    elif device == 'gpu':
         os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU:0'
         return tflite.Interpreter(model_path=model_path)
    else:
        raise ValueError("Invalid device. Choose either 'cpu', 'tpu', or 'gpu'.")

def main():
    # Parse command line arguments
    args = parse_args()

    # Load the label file
    with open(args.label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the interpreter
    interpreter = load_interpreter(args.model_path, args.device)

    # Allocate tensors
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Define the video capture device
    cap = cv2.VideoCapture(args.video_device)

    # Set the video capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    while True:
        # Capture a frame
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        resized_frame = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
        input_data = image_processing.reshape_and_rescale(
            np.array(resized_frame), input_details[0]['shape'][1:3]).astype(input_details[0]['dtype'])

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get the index of the highest confidence score
        top_index = np.argmax(output_data)

        # Get the corresponding label and confidence score
        label = labels[top_index]
        score = output_data[0, top_index]

        # Draw the label and confidence score on the frame
        label_text = f"{label}: {score:.2f}"
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Frame', frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
