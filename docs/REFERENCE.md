# SoundMap setup reference

This guide covers the retained application setup. [Source and license scope](../THIRD_PARTY_NOTICES.md) documents its component license boundaries.

This is a research/test implementation. Live acquisition, camera overlays and model inference require separate verification.

This project aims to create an acoustic camera that can visualize sound sources in a room.

## Overview

Here is a brief overview of the project structure:

![alt text](../overview.png "Title")

## Requirements

### Hardware

- Microphone Array
- USB Camera (optional)

### Python packages

- Python == 3.11
- Acoular
- TensorFlow and a model checkpoint
- OpenCV (optional)
- Flask (optional)

## Environment

1. Create a new environment: <br> `conda create -n acoustic_camera python=3.11`

2. Activate the environment: <br> `conda activate acoustic_camera`

3. Install the required libraries: <br> `pip install -r requirements.txt`

## Usage

- Clone this repository and navigate into the `acoustic-camera` folder.

- To run with default settings, use: `python start.py`.

- To run without video output, use the `--no-flask` flag.

- To use a specific model, add the flag `--model path/to/model/folder`.
