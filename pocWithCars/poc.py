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
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

TESTED_SUBDIR_PATH = "TT_DB/white"
DEST_PATH = "data"

BATCH_SIZE = 8
INPUT_SHAPE = (300,300,3)
NUM_CLASSES = 4
EPOCHS = 5

def preprocessVariableSet(usedPercentage: float, dbPath: str):
    """
    Function that takes makes changes to the dataset based on args

    Args:
        usedPercentage: what percentage of tested subset should be used in this experiment
        dbPath: path to dir containing data
    """
    sourcePath = os.path.join(dbPath, TESTED_SUBDIR_PATH)
    trainPath = os.path.join(dbPath, DEST_PATH, "train", "white")
    valPath = os.path.join(dbPath, DEST_PATH, "val", "white")

    if os.path.exists(trainPath):
        shutil.rmtree(trainPath)

    if os.path.exists(valPath):
        shutil.rmtree(valPath)

    os.makedirs(trainPath)
    os.makedirs(valPath)

    images = [img for img in os.listdir(sourcePath) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    num_images = len(images)

    random.shuffle(images)

    train_split = int(0.6 * num_images)
    val_split = int(0.2 * num_images)
    test_split = int(0.2 * num_images)  # Smaller test set

    all_images = images[:train_split + val_split + test_split]

    random.shuffle(all_images)  # Shuffle the images

    train_images = all_images[:ceil(train_split*usedPercentage)]
    val_images = all_images[ceil(train_split*usedPercentage):ceil(train_split*usedPercentage) + ceil(val_split*usedPercentage)]

    for img in tqdm(train_images):
        shutil.copy2(os.path.join(sourcePath, img), trainPath)

    for img in tqdm(val_images):
        shutil.copy2(os.path.join(sourcePath, img), valPath)

def getLoaders(dbPath: str):
    """
    Create tf dataset objects 

    Args:
        dbPath: path to preprocessed db
    
    Returns:
        - dataset object for train split
        - dataset object for val split
        - dataset object for test split
    """
    trainDir = os.path.join(dbPath, "train")
    valDir = os.path.join(dbPath, "val")
    testDir = os.path.join(dbPath, "test")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        trainDir,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=(300, 300),
        shuffle=True,
        seed=321
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        valDir,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=(300, 300),
        shuffle=False
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        testDir,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=(300, 300),
        shuffle=False
    )

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset

def getModel(inputShape: Tuple, numClasses: int):
    """
    Create a simple tf model

    Args:
        inputShape: data shape
        numClasses: how many classes are there
    
    Returns:
        a tf model
    """
    baseModel = ResNet50(weights="imagenet", include_top=False, input_shape=inputShape)

    x = GlobalAveragePooling2D()(baseModel.output)
    x = Dense(258)(x)

    output = Dense(numClasses, activation="softmax")(x)

    model = Model(inputs=baseModel.input, outputs=output)

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

    trainDataset, valDataset, testDataset = getLoaders(os.path.join(args.dbPath, "data"))

    model = getModel(INPUT_SHAPE, NUM_CLASSES)

    modelsPath = os.path.join(os.getcwd(), "models")
    if os.path.exists(modelsPath):
        shutil.rmtree(modelsPath)
    os.makedirs(modelsPath)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[tf.keras.metrics.F1Score(average="macro")])
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(modelsPath, 'model_epoch{epoch:02d}.h5'),
        monitor='val_f1score',
        save_best_only=False,
        mode='max',
        verbose=1
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

    metricsPath = os.path.join(os.getcwd(), "results")
    os.makedirs(metricsPath, exist_ok=True)

    saveDict = {
        "train_f1_score": round(train_f1_scores[best_epoch],3),
        "val_f1_score": round(val_f1_scores[best_epoch],3),
        "test_f1_score": round(test_f1_score,3),
        "test_loss": round(test_loss,3)
    }

    jsonPath = os.path.join(metricsPath, f"testSplit{args.testedDataShare}.json")
    if os.path.exists(jsonPath):
        os.remove(jsonPath)

    with open(jsonPath, 'w') as f:
        json.dump(saveDict, f, indent=2)
