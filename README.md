## 🐾 Cat Eating Monitor: For my cat Sesame ❤️
This project applies computer vision and machine learning to monitor my cat Sesame, through video footage captured by our Blink home security camera. This project identifies when she's present and analyzes her feeding behavior over time.

As a university student away from home, I really miss Sesame and want a reliable way to keep tabs on her eating habits without relying on sporadic camera notifications. The vet mentioned that she's a bit chubby too, so I needed an easy way to track her eating habits.

![frame_44-cat-bowl-2025-05-16t01-12-33-00-00_00000](https://github.com/user-attachments/assets/88a74998-df7e-4568-9a77-3e47f647dee1)

## ✨ Project Highlights
- Automated Video Retrieval: Connects to Blink cameras via the blinkpy API to pull historical video clips securely.
- Efficient Sampling & Labeling: Samples raw clips for quick manual review and labeling during the initial training phase.
- Video Processing & Frame Extraction: Uses OpenCV to extract and manage frames for training and inference.
- Transfer Learning with PyTorch: Fine-tunes pre-trained models like ResNet for accurate classification on a small, custom dataset.
- Organized Data Pipeline: Automatically sorts and stores clips and frame data into meaningful categories using OpenCV.
- Credential Management: Sensitive credentials are stored securely in a local .json file and excluded from version control with .gitignore.
 
![Screenshot 2025-06-19 230115](https://github.com/user-attachments/assets/8d2d899d-93a4-4ded-ab04-a02dce28ef23)

###### Two-Stage Classification Pipeline
1. Cat Detection: A lightweight model filters out clips with no cat or irrelevant motion (e.g., people, shadows).
2. Eating Recognition: A specialized model will detect and classify eating behavior (e.g., duration, frequency) from cat-present clips.


## Tech Stack & Skills
- Language: Python
- Async Programming: asyncio
- Hardware Integration: blinkpy API for interfacing with Blink cameras
- Computer Vision: OpenCV for frame extraction and preprocessing
- Deep Learning: PyTorch for training and deploying classification models
- Data Handling: NumPy, os, shutil for file and data operations
- Annotation Strategy: Manual image labeling of my cat's activity for supervised learning
- Version Control: Git and GitHub for source tracking and documentation

## Local web monitor

Install the dependencies in `requirements.txt` in your Python environment. Start
the existing `setup.py` once to authenticate with Blink and save the local
`blink_config.json` file. Then run `python web_app.py` and open
<http://127.0.0.1:8000>. The dashboard can classify an uploaded image and, after
connecting to Blink and selecting a camera, poll for new motion clips about once
a minute. New clips are saved under `data/blink-clips/` and sampled for cat
presence locally.

Open the dashboard from the Python server URL above, not from VS Code Live
Server or another static preview; the frontend needs the API served by
`web_app.py` on the same origin.

The dashboard binds only to the local machine. Keep `blink_config.json` private.
The saved ResNet18 checkpoint is the PyTorch model trained on this repository's
own bowl-camera frames. Its positive examples are labeled `cat-present` and its
negative examples `no-cat`, so the dashboard treats the positive class as
**Sesame likely present** in that familiar scene. It has not been trained with
other cats as negative examples and cannot reliably distinguish Sesame from a
different cat. The reported 95.7% validation accuracy is also optimistic:
frames from 105 clips occur in both the training and validation splits. Collect
and label other-cat examples and hold out whole clips for a reliable
identity-specific evaluation before relying on it to distinguish individual
cats. To fine-tune the existing checkpoint on the current labels using only
clip-disjoint validation frames, run `python train_model.py`, then run
`python evaluate.py`. Only 18 validation frames from 10 clips remain after
removing overlapping clips, so treat that score as preliminary.

BlinkPy provides saved motion clips here, not a live video stream, so detection
happens after a clip is available to sync.
