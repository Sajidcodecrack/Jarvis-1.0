import cv2
import os
from PIL import Image
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
ids = []

image_paths = [os.path.join('dataset', f) for f in os.listdir('dataset')]

for image_path in image_paths:
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_np = np.array(img, 'uint8')
    faces.append(img_np)
    ids.append(1)  # Use your unique ID

recognizer.train(faces, np.array(ids))
recognizer.save('trainer.yml')
