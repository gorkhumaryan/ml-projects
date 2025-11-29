#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import sklearn as sk
from PIL import Image
import os
#%%
data_dir = r"C:\docs\processed_data"

label_map = {
    "angry": 0,
    "surprise": 1
}

filepaths = []
labels = []

for label in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, label)
    for fname in os.listdir(class_dir):
        filepaths.append(os.path.join(class_dir, fname))
        labels.append(label)

X_train, X_temp, y_train, y_temp = sklearn.model_selection.train_test_split(
    filepaths, labels, test_size=0.3, random_state=42)

X_val, X_test, y_val, y_test = sklearn.model_selection.train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

y_train = [label_map[label] for label in y_train]
y_val = [label_map[label] for label in y_val]
y_test = [label_map[label] for label in y_test]

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image = image/255.0
    return image, label

train_ds = train_ds.map(load_and_preprocess_image)
val_ds = val_ds.map(load_and_preprocess_image)
test_ds = test_ds.map(load_and_preprocess_image)

train_ds = train_ds.shuffle(2000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)



model = tf.keras.Sequential([
    tf.keras.Input(shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=10)

model.summary()


plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history["val_loss"], label='val_loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

model.save("my_cnn_model.h5")