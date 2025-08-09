import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load images and labels
def load_dataset(data_path, img_size=(100, 100)):
    images = []
    labels = []
    for person_name in os.listdir(data_path):
        person_path = os.path.join(data_path, person_name)
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(person_name)
    return np.array(images), np.array(labels)

# Dataset path
dataset_path = "dataset"
X, y = load_dataset(dataset_path)
print("Data loaded:", X.shape, y.shape)

# Normalize
X = X.astype('float32') / 255.0

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y)), activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy: {:.2f}%".format(acc * 100))

# Predict on a single image
def recognize_face(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (100, 100))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_label = le.inverse_transform([class_index])[0]
    return class_label

# Example test
test_img_path = "dataset/person1/img1.jpg"
result = recognize_face(test_img_path)
print("Recognized as:", result)
