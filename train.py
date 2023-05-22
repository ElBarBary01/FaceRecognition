import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import matplotlib as plt
print(tf.__version__)
face_detector = MTCNN()

# Load the class names
lfw_data_dir = "./lfw"
lfw_data = tf.keras.preprocessing.image_dataset_from_directory(lfw_data_dir, labels="inferred", label_mode="int")
class_names = lfw_data.class_names
print(class_names)
def detect_faces(img):
    faces = face_detector.detect_faces(img)
    face_rects = [(face['box'][0], face['box'][1], face['box'][2], face['box'][3]) for face in faces]
    return face_rects

# Function to recognize faces in an image
def recognize_faces(img, model):
    faces = detect_faces(img)
    recognized_faces = []

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_img_resized = cv2.resize(face_img, (224, 224))
        face_img_preprocessed = preprocess_input(face_img_resized)
        face_img_expanded = np.expand_dims(face_img_preprocessed, axis=0)

        preds = model.predict(face_img_expanded)
        label = np.argmax(preds)
        name = class_names[label]
        confidence = np.max(preds)

        recognized_faces.append((name, confidence, (x, y, w, h)))

    return recognized_faces

# Load the trained Facenet model
facenet = tf.keras.models.load_model('facenet_model.h5')

# Read an input image
input_image = cv2.imread('Omar_Sharif_0002.jpg')

# Recognize faces in the input image
recognized_faces = recognize_faces(input_image, facenet)

# Draw rectangles and labels around recognized faces
for (name, confidence, (x, y, w, h)) in recognized_faces:
    cv2.rectangle(input_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(input_image, f"{name}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Save the output image
cv2.imwrite('./output_image.png', input_image)

# Show the output image
plt.imshow('Final', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
