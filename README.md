# AI Rock-Paper-Scissors Robot

## Project Overview

This project was developed as a final project for the Artificial Intelligence for Robotics course.

The system uses a pre-trained neural network from MediaPipe Hands to recognize hand gestures in real time through a webcam. The recognized gesture is classified as Rock, Paper, or Scissors and used to play against an AI robot.

The robot combines real-time gesture recognition with a prediction module based on a Markov Chain model to analyze player behavior and improve decision-making.

---

## Features

* Real-time hand tracking using MediaPipe Hands.
* Rock, Paper, and Scissors gesture recognition.
* AI robot opponent.
* Markov Chain-based prediction system.
* Performance metrics and statistics.
* Interactive graphical user interface using Pygame.
* Real-time webcam integration using OpenCV.

---

## Technologies Used

* Python
* OpenCV
* MediaPipe
* Pygame
* TensorFlow Lite (via MediaPipe)

---

## System Architecture

Camera Input
→ MediaPipe Hands
→ Gesture Classification
→ Markov Prediction Module
→ Game Logic
→ Robot Response

---

## How to Run

### Install Requirements

```bash
pip install pygame opencv-python mediapipe
```

### Run the Project

```bash
python rps_robot.py
```

---

## AI Component

The project uses MediaPipe Hands, a pre-trained deep learning model, for real-time hand detection and landmark extraction.

A Markov Chain predictor is used to analyze previous player moves and estimate future actions.

---

## Project Authors

* Hamza Helo
* [Partner Name]

Artificial Intelligence for Robotics
Final Project – 2026

---

## Repository

This repository contains the complete source code, documentation, and project materials.
