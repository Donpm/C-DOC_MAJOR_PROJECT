import os
import cv2
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def contrast_stretching(img):
    # Convert image to grayscale if it is in color
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply contrast stretching
    min_val, max_val, _, _ = cv2.minMaxLoc(img)
    stretched_img = np.uint8((img - min_val) / (max_val - min_val) * 255)

    # If the image was originally color, stack it back to 3 channels
    if len(stretched_img.shape) == 2:
        stretched_img = cv2.cvtColor(stretched_img, cv2.COLOR_GRAY2BGR)

    return stretched_img

def load_and_preprocess_data(folder_path):
    dataset = []
    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)
        img = cv2.imread(img_path)

        # Apply contrast stretching
        img_stretched = contrast_stretching(img)

        img_stretched = cv2.resize(img_stretched, (224, 224))
        img_stretched = preprocess_input(img_stretched)
        dataset.append(img_stretched)
    return dataset

# Instantiate ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add your custom layers on top of ResNet50
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Add dropout to prevent overfitting
model.add(Dense(1, activation='sigmoid'))

# Fine-tune some layers if needed
for layer in model.layers[0].layers[:-50]:
    layer.trainable = False

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.001),  # Adjust the learning rate
              metrics=['accuracy'])

# Load and split the data
healthy_cow_dataset = load_and_preprocess_data('/content/drive/MyDrive/Proj_dset/hcaugmented-20231027T034326Z-001/hcaugmented')
infected_cow_dataset = load_and_preprocess_data('/content/drive/MyDrive/Proj_dset/lcaugmented-20231027T034148Z-001/lcaugmented')

X = np.array(healthy_cow_dataset + infected_cow_dataset)
y = np.concatenate([np.zeros(len(healthy_cow_dataset)), np.ones(len(infected_cow_dataset))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add validation_split for monitoring validation accuracy
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

model.fit(X_train, y_train, epochs=20, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

# Evaluate on the test set
predictions = model.predict(X_test)
y_pred = (predictions > 0.5).astype(int).flatten()  # Adjust the threshold based on your task
test_accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", test_accuracy)

# Save the model
model.save('/content/drive/MyDrive/Proj_dset/model/Contrast_Stretching_Resnet.h5')
