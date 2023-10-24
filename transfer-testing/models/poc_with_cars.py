import os
import random
import shutil
import json
import numpy as np

from argparse import ArgumentParser
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import (
    Dense,
    Flatten,
)
from tensorflow.keras.models import Sequential

# Experiment names: book1, book2, number of added classes, percentage
# of added classes content.

Experiment_Name = "AB_3_100"

USED_GPU = "GPU:0"

BATCH_SIZE = 32
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 35
EPOCHS = 100


def getLoaders(dbPath: str):
    trainDir = os.path.join(dbPath, "train")
    valDir = os.path.join(dbPath, "val")
    testDir = os.path.join(dbPath, "test")
    testTTDir = os.path.join(dbPath, "test_TT")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        trainDir,
        labels="inferred",
        label_mode="categorical",
        batch_size=BATCH_SIZE,
        image_size=(224, 224),
        shuffle=True,
        seed=321,
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        valDir,
        labels="inferred",
        label_mode="categorical",
        batch_size=BATCH_SIZE,
        image_size=(224, 224),
        shuffle=False,
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        testDir,
        labels="inferred",
        label_mode="categorical",
        batch_size=BATCH_SIZE,
        image_size=(224, 224),
        shuffle=False,
    )

    testTT_dataset = tf.keras.utils.image_dataset_from_directory(
        testTTDir,
        labels="inferred",
        label_mode="categorical",
        batch_size=BATCH_SIZE,
        image_size=(224, 224),
        shuffle=False,
    )

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    testTT_dataset = testTT_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset, testTT_dataset


def getModel(inputShape: Tuple, numClasses: int):
    model = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=INPUT_SHAPE,
        classes=numClasses,
        classifier_activation="softmax",
    )

    return model


def preprocess(images, labels):
    return preprocess_input(images), labels


if __name__ == "__main__":
    random.seed(2137)

    parser = ArgumentParser()

    parser.add_argument(
        "--dbPath",
        dest="dbPath",
        type=str,
        default="BookDataset(A_B)",
        help="Path to dir with raw subset we are interested in",
    )

    args = parser.parse_args()

    trainDataset, valDataset, testDataset, testTTDataset = getLoaders(
        args.dbPath
    )

    base_model = VGG16(
        weights="imagenet", include_top=False, input_shape=INPUT_SHAPE
    )
    base_model.trainable = False  # Not trainable weights

    trainDataset = trainDataset.map(preprocess)
    valDataset = valDataset.map(preprocess)

    flatten_layer = Flatten()
    dense_layer_1 = Dense(50, activation="relu")
    dense_layer_2 = Dense(20, activation="relu")
    prediction_layer = Dense(NUM_CLASSES, activation="softmax")

    model = Sequential(
        [
            base_model,
            flatten_layer,
            dense_layer_1,
            dense_layer_2,
            prediction_layer,
        ]
    )

    # model = getModel(INPUT_SHAPE, NUM_CLASSES)

    modelsPath = os.path.join(os.getcwd(), "models", f"{Experiment_Name}")
    if os.path.exists(modelsPath):
        shutil.rmtree(modelsPath)
    os.makedirs(modelsPath)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(modelsPath, "model_epoch{epoch:02d}.h5"),
        monitor="val_accuracy",
        save_best_only=False,
        mode="max",
        verbose=1,
    )

    epochs = EPOCHS
    with tf.device(USED_GPU):
        history = model.fit(
            trainDataset,
            epochs=epochs,
            validation_data=valDataset,
            callbacks=[checkpoint_callback],
        )

    train_accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    best_epoch = np.argmax(val_accuracy)
    best_model = tf.keras.models.load_model(
        os.path.join(modelsPath, f"model_epoch{best_epoch:02d}.h5")
    )

    with tf.device(USED_GPU):
        test_loss, test_accuracy = best_model.evaluate(testDataset)
        testTT_loss, testTT_accuracy = best_model.evaluate(testTTDataset)

    metricsPath = os.path.join(os.getcwd(), "results", Experiment_Name)
    os.makedirs(metricsPath, exist_ok=True)

    saveDict = {
        "train_accuracy": round(train_accuracy[best_epoch], 3),
        "val_accuracy": round(val_accuracy[best_epoch], 3),
        "test_accuracy": round(test_accuracy, 3),
        "test_loss": round(test_loss, 3),
        "testTT_accuracy": round(testTT_accuracy, 3),
        "testTT_loss": round(testTT_loss, 3),
    }

    jsonPath = os.path.join(
        metricsPath, f"trained_TT_Classes{Experiment_Name}.json"
    )
    if os.path.exists(jsonPath):
        os.remove(jsonPath)

    with open(jsonPath, "w") as f:
        json.dump(saveDict, f, indent=2)
