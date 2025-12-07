import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import sklearn as sk
from PIL import Image
import os

from tensorflow.python.keras.callbacks import EarlyStopping

def main():
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
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )
    
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
        tf.keras.layers.Dropout(0.4),
    
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[early_stop]
    )
    
    # fig, ax = plt.subplots(2, 1, figsize=(8, 10))
    #
    # # ax[0].plot(history.history['loss'], label='train_loss')
    # # ax[0].plot(history.history["val_loss"], label='val_loss')
    # # ax[0].set_title('Loss')
    # # ax[0].set_ylabel('Loss')
    # # ax[0].set_xlabel('Epoch')
    # # ax[0].legend()
    # #
    # # ax[1].plot(history.history['accuracy'], label='train_acc')
    # # ax[1].plot(history.history["val_accuracy"], label='val_acc')
    # # ax[1].set_title('Accuracy')
    # # ax[1].set_ylabel('Accuracy')
    # # ax[1].set_xlabel('Epoch')
    # # ax[1].legend()
    #
    # plt.tight_layout()
    # plt.show()
    test_loss, test_acc = model.evaluate(test_ds)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)
    
    predictions = model.predict(test_ds)
    predicted_labels = (predictions > 0.5).astype(int)
    
    reverse_map = {0: "angry", 1: "surprise"}
    
    for images, labels in test_ds.take(1):
        preds = model.predict(images)
        preds = (preds > 0.5).astype(int)   # shape: (batch, 1)
    
        for i in range(len(images)):
            true_label = int(labels[i].numpy())
            pred_label = int(preds[i][0])   # CONVERT TO INT
    
            plt.imshow(images[i].numpy())
            plt.title(f"True: {reverse_map[true_label]}, Pred: {reverse_map[pred_label]}")
            plt.axis("off")
            plt.show()



    model.save("models/saved_model.keras")

if __name__ = '__main__':
    main()

