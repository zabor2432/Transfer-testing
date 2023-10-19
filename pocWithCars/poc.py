import os
import random
import shutil
import json
import numpy as np

from argparse import ArgumentParser
from tqdm import tqdm
from math import ceil
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D
from tensorflow.keras.models import Model

TESTED_SUBDIR_PATH = "TT_DB/white"
DEST_PATH = "data"

BATCH_SIZE = 32
INPUT_SHAPE = (64,64,3)
NUM_CLASSES = 4
EPOCHS = 10

def resize_image(image, label):
    # Resize the image to the desired dimensions
    resized_image = tf.image.resize(image, (64, 64))
    return resized_image, label

def preprocessVariableSet(usedPercentage: float, dbPath: str):
    """
    Function that makes changes to the dataset based on args

    Args:
        usedPercentage: what percentage of tested subset should be used in this experiment
        dbPath: path to dir containing data
    """
    sourcePath = os.path.join(dbPath, TESTED_SUBDIR_PATH)
    trainPath = os.path.join(dbPath, DEST_PATH, "train", "white")

    if os.path.exists(trainPath):
        shutil.rmtree(trainPath)   

    os.makedirs(trainPath)

    images = [img for img in os.listdir(sourcePath) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    num_images = len(images)

    random.shuffle(images)

    train_split = int(0.6 * num_images)
    val_split = int(0.2 * num_images)
    test_split = int(0.2 * num_images)  # Smaller test set

    all_images = images[:train_split + val_split + test_split]

    random.shuffle(all_images)  # Shuffle the images

    train_images = all_images[:ceil(train_split*usedPercentage)]

    for img in tqdm(train_images):
        shutil.copy2(os.path.join(sourcePath, img), trainPath)


def getLoaders(dbPath: str):
    """
    Create tf dataset objects 

    Args:
        dbPath: path to preprocessed db
    
    Returns:
        - dataset object for train split
        - dataset object for val split
        - dataset object for test split
        - dataset object for testTT split
    """
    trainDir = os.path.join(dbPath, "train")
    valDir = os.path.join(dbPath, "val")
    testDir = os.path.join(dbPath, "test")
    testTTDir = os.path.join(dbPath, "testTT")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        trainDir,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=(300, 300),
        shuffle=True,
        seed=321
    )

    train_dataset = train_dataset.map(resize_image)

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        valDir,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=(300, 300),
        shuffle=False
    )

    val_dataset = val_dataset.map(resize_image)

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        testDir,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=(300, 300),
        shuffle=False
    )

    test_dataset = test_dataset.map(resize_image)

    testTT_dataset = tf.keras.utils.image_dataset_from_directory(
        testTTDir,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=(300, 300),
        shuffle=False
    )

    testTT_dataset = testTT_dataset.map(resize_image)

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    testTT_dataset = testTT_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset, testTT_dataset

def getModel(inputShape: Tuple, numClasses: int):
    """
    Create a simple tf model

    Args:
        inputShape: data shape
        numClasses: how many classes are there
    
    Returns:
        a tf model
    """
    model = tf.keras.Sequential([
        Conv2D(64, (3,3), padding="same", activation="relu", input_shape=inputShape),
        Conv2D(64, (3,3), padding="same", activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(numClasses, activation = "softmax")
    ])

    return model

if __name__ == "__main__":
    random.seed(2137)
    parser = ArgumentParser()

    parser.add_argument('--testedDataShare', dest='testedDataShare', type=float, default=0.1,
                        help="what percentage of tested subset should be added to train and val splits")
    
    parser.add_argument("--dbPath", dest="dbPath", type=str, default="Transfer_testing_db",
                        help="Path to dir with raw subset we are interested in")
    
    args = parser.parse_args()

    preprocessVariableSet(args.testedDataShare, args.dbPath)

    trainDataset, valDataset, testDataset, testTTDataset = getLoaders(os.path.join(args.dbPath, "data"))

    model = getModel(INPUT_SHAPE, NUM_CLASSES)

    modelsPath = os.path.join(os.getcwd(), "models", f"test_share{args.testedDataShare}")
    if os.path.exists(modelsPath):
        shutil.rmtree(modelsPath)
    os.makedirs(modelsPath)

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=[tf.keras.metrics.F1Score(average="macro")])
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(modelsPath, 'model_epoch{epoch:02d}.h5'),
        monitor='val_f1score',
        save_best_only=False,
        mode='max',
        verbose=1,
        #period=5
    )

    epochs = EPOCHS
    with tf.device("GPU:0"):
        history = model.fit(trainDataset, epochs=epochs, validation_data=valDataset, callbacks=[checkpoint_callback])

    train_f1_scores = history.history['f1_score']
    val_f1_scores = history.history['val_f1_score']

    best_epoch = np.argmax(val_f1_scores)
    best_model = tf.keras.models.load_model(os.path.join(modelsPath, f'model_epoch{best_epoch:02d}.h5'))

    with tf.device("GPU:0"):
        test_loss, test_f1_score = best_model.evaluate(testDataset)
        testTT_loss, testTT_f1_score = best_model.evaluate(testTTDataset)

    metricsPath = os.path.join(os.getcwd(), "results")
    os.makedirs(metricsPath, exist_ok=True)

    saveDict = {
        "train_f1_score": round(train_f1_scores[best_epoch],3),
        "val_f1_score": round(val_f1_scores[best_epoch],3),
        "test_f1_score": round(test_f1_score,3),
        "test_loss": round(test_loss,3),
        "testTT_f1_score": round(testTT_f1_score,3),
        "testTT_loss": round(testTT_loss,3)
    }

    jsonPath = os.path.join(metricsPath, f"testSplit{args.testedDataShare}.json")
    if os.path.exists(jsonPath):
        os.remove(jsonPath)

    with open(jsonPath, 'w') as f:
        json.dump(saveDict, f, indent=2)
