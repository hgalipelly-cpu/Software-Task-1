# Autonomous Spatial Tracking and Guidance System

## Objective

This project implements a real-time ArUco marker tracking system using OpenCV and Python. The system detects an ArUco marker through a webcam, calculates its centroid, and provides movement guidance based on the marker's position relative to the center of the frame.

## Features

* Real-time ArUco marker detection
* Center crosshair display
* Dynamic error vector visualization
* Movement commands:

  * MOVE LEFT
  * MOVE RIGHT
  * MOVE UP
  * MOVE DOWN
  * APPROACH
* LOCK ENGAGED indication when the marker is centered
* TARGET LOST indication when the marker leaves the camera view

## Requirements

* Python 3.x
* OpenCV (opencv-contrib-python)
* NumPy

## Installation

pip install -r requirements.txt

## Running the Project

python main.py

## Working

The webcam continuously captures video frames. The ArUco marker is detected and its centroid is calculated. An error vector is drawn from the center of the frame to the marker centroid. Based on the positional error, movement instructions are displayed. When the marker is within the specified threshold around the center, the system displays "LOCK ENGAGED".
