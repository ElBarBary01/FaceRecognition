import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
# from google.colab.patches import cv2_imshow

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to detect faces in an image
def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Load the LFW dataset
lfw_data_dir = "./lfw"
lfw_data = tf.keras.preprocessing.image_dataset_from_directory(lfw_data_dir, labels="inferred", label_mode="categorical")

# Preprocess the images and labels
X = []
y = []

for images, labels in lfw_data:
    for image, label in zip(images, labels):
        faces = detect_faces(image.numpy().astype(np.uint8))
        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            X.append(face_img)
            y.append(label.numpy())

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for Facenet
def preprocess(x):
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    return x

# Create the Facenet model using MobileNetV2 as the base model
input_shape = (224, 224, 3)
base_model = MobileNetV2(input_shape=input_shape, include_top=False, pooling='avg')
inputs = Input(shape=input_shape)
x = Lambda(preprocess)(inputs)
x = base_model(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(len(np.unique(y)), activation='sigmoid')(x)
facenet = Model(inputs, outputs)

# Compile the model
facenet.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                             zoom_range=0.2, horizontal_flip=True)

# Train the model
facenet.fit(datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), epochs=10)

# Save the model
facenet.save('facenet_model.h5')

# Function to recognize faces in an image
def recognize_faces(img, model):
    faces = detect_faces(img)
    face_imgs = []
    face_rects = []

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_imgs.append(face_img)
        face_rects.append((x, y, w, h))

    face_imgs = np.array(face_imgs)
    preds = model.predict(face_imgs)

    recognized_faces = []
    for pred, rect in zip(preds, face_rects):
        label = np.argmax(pred)
        confidence = np.max(pred)
        recognized_faces.append((label, confidence, rect))

    return recognized_faces

# Load the trained Facenet model
facenet = tf.keras.models.load_model('facenet_model.h5')

# Read an input image
input_image = cv2.imread('./people.png')

# Recognize faces in the input image
recognized_faces = recognize_faces(input_image, facenet)

# Draw rectangles and labels around recognized faces
for (label, confidence, (x, y, w, h)) in recognized_faces:
    cv2.rectangle(input_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(input_image, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Save the output image
cv2.imwrite('./', input_image)

# Show the output image
cv2.imshow(input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()