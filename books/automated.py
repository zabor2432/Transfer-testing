import os
import random
import shutil
import json
import numpy as np
import datetime

from argparse import ArgumentParser
from typing import Tuple

book_paths = {
    'A': '/local_storage/gwo/public/transfer-testing/books/CB/978-83-7420-600-6',
    'B': '/local_storage/gwo/public/transfer-testing/books/CB/978-83-7420-629-7',
    'C': '/local_storage/gwo/public/transfer-testing/books/CB/978-83-7420-601-3',
    'D': '/local_storage/gwo/public/transfer-testing/books/CB/978-83-7420-603-7',
    'E': '/local_storage/gwo/public/transfer-testing/books/CB/978-83-7420-647-1',
    'G': '/local_storage/gwo/public/transfer-testing/books/CB/978-83-7420-648-8',
    'H': '/local_storage/gwo/public/transfer-testing/books/CB/978-83-7420-754-6'
}

# python3 automated.py --Used_GPU x --Base_Class y --Added_Class z --Extra_Name aaa

BATCH_SIZE = 32
INPUT_SHAPE = (224, 224, 3)
EPOCHS = 10

Base_Classes = 32
num_added_classes = [3, 10, 15, 20]
ex_percentage = [100, 50, 25, 10]
del_percentage = [50, 50, 40, 0]

def getLoaders(dbPath: str):
    trainDir = os.path.join(dbPath, "train")
    valDir = os.path.join(dbPath, "val")
    testDir = os.path.join(dbPath, "test")
    testTTDir = os.path.join(dbPath, "test_T")

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

def book_prepare(num_folders, destination):    

    num_folders_to_copy = num_folders

    d= f'/home/macierz/s181655/projektBadawczy/Transfer-testing/Books/{destination}'


    # for i, class_folder in enumerate(sorted(os.listdir(os.path.join(added_class, 'train')))):
    #     if i >= 0:  # Start from the third element (index 2)
    #         if i - 2 < num_folders_to_copy:  # Limit the number of folders to copy
    #             source_class_path = os.path.join(added_class, 'train', class_folder)
    #             target_class_folder = class_folder + '_TT'
    #             target_class_path = os.path.join(d, 'train', target_class_folder)

    #             shutil.copytree(source_class_path, target_class_path)
    #         else:
    #             break
    for i, class_folder in enumerate(sorted(os.listdir(os.path.join(added_class, 'train')))):
        if i >= 0:  # Start from the third element (index 2)
            if i < num_folders_to_copy:  # Limit the number of folders to copy
                source_class_path = os.path.join(added_class, 'train', class_folder)
                target_class_folder = class_folder + '_TT'
                target_class_path = os.path.join(d, 'train', target_class_folder)

                shutil.copytree(source_class_path, target_class_path)
            else:
                break


    for i, class_folder in enumerate(sorted(os.listdir(os.path.join(added_class, 'val')))):
        if i >= 0:  # Start from the third element (index 2)
            if i < num_folders_to_copy:  # Limit the number of folders to copy
                source_class_path = os.path.join(added_class, 'val', class_folder)
                target_class_folder = class_folder + '_TT'
                target_class_path = os.path.join(d, 'val', target_class_folder)

                shutil.copytree(source_class_path, target_class_path)
            else:
                break

    for i, class_folder in enumerate(sorted(os.listdir(os.path.join(added_class, 'test')))):
        if i >= 0:  # Start from the third element (index 2)
            if i  < num_folders_to_copy:  # Limit the number of folders to copy
                source_class_path = os.path.join(added_class, 'test', class_folder)
                target_class_folder = class_folder + '_TT'
                target_class_path = os.path.join(d, 'test', target_class_folder)

                shutil.copytree(source_class_path, target_class_path)
            else:
                break

    for i, class_folder in enumerate(sorted(os.listdir(os.path.join(added_class, 'test')))):
        if i >= 0:  # Start from the third element (index 2)
            if i  < num_folders_to_copy:  # Limit the number of folders to copy
                source_class_path = os.path.join(added_class, 'test', class_folder)
                target_class_folder = class_folder + '_TT'
                target_class_path = os.path.join(d, 'test_T', target_class_folder)

                shutil.copytree(source_class_path, target_class_path)
            else:
                break

def dir_delete(destination):
    root_path = f"/home/macierz/s181655/projektBadawczy/Transfer-testing/Books/{destination}"
    for root, dirs, files in os.walk(root_path):
        for dir_name in dirs:
            if dir_name.endswith('_TT'):
                dir_path = os.path.join(root, dir_name)
                shutil.rmtree(dir_path)
                print(f"Deleted directory: {dir_path}")

def tt_percentage(x, destination):
   
    percentage_to_delete = x
    root_path = f'/home/macierz/s181655/projektBadawczy/Transfer-testing/Books/{destination}/train'

    # Walk through the directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_path):
        for dirname in dirnames:
            if dirname.endswith('_TT'):
                # Calculate the number of files to delete based on the percentage
                files_to_delete = int(len(os.listdir(os.path.join(dirpath, dirname)) * percentage_to_delete) / 100)

                # Get a list of files in the directory
                files_in_directory = sorted(os.listdir(os.path.join(dirpath, dirname)))

                # Determine the starting index for deletion
                start_index = 0

                # Delete the selected files
                files_to_delete_list = files_in_directory[start_index:start_index + files_to_delete]

                # Delete the selected files
                for file_to_delete in files_to_delete_list:
                    file_path = os.path.join(dirpath, dirname, file_to_delete)
                    os.remove(file_path)

def log_error(error_msg):
    from datetime import datetime
    error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_filename = f"/home/macierz/s181655/projektBadawczy/Transfer-testing/Books/Errors/error_log_{error_time}.txt"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    with open(log_filename, 'w') as log_file:
        log_file.write(error_msg)
       
if __name__ == "__main__":
        
    parser = ArgumentParser()

    parser.add_argument(
        "--Used_GPU",
        dest="Used_GPU",
        type=int,
        default=0,
        help="choose gpu to rune code on",
    )
    parser.add_argument(
        "--Base_Class",
        dest="Base_Class",
        type=str,
        help="choose base class",
    )
    
    parser.add_argument(
        "--Added_Class",
        dest="Added_Class",
        type=str,
        help="choose added (TT) class",
    )
    
    parser.add_argument(
        "--Extra_Name",
        dest="Extra_Name",
        type=str,
        help="choose gpu to rune code on",
        default=None,
        required=False
    )

    args = parser.parse_args()
    
    USED_GPU = f"GPU:{args.Used_GPU}"
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.Used_GPU}"
    
    base_class = book_paths[args.Base_Class]
    added_class = book_paths[args.Added_Class]
    dest_folder = f'BookDataset{args.Base_Class}_{args.Added_Class}'
    experiment_prefix = f"{args.Extra_Name}_{args.Base_Class}{args.Added_Class}"
    #experiment_prefix = f"{args.Base_Class}{args.Added_Class}"
    dir_delete(dest_folder)

    import tensorflow as tf
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.layers import (
        Dense,
        Flatten,
    )
    from tensorflow.keras.models import Sequential
    from tensorflow.python.client import device_lib
    
    for x in num_added_classes:
        
        try:
            book_prepare(x, dest_folder)
        except Exception as ex:
            error_message = f"An error occurred in experiment: {experiment_prefix}_{x} while book_prepare {x}: {ex}\n"
            print(error_message)
            log_error(error_message)
            
        for (y, z) in zip(ex_percentage, del_percentage):
    
            try:
                random.seed(2137)

                Experiment_Name = f"{experiment_prefix}_{x}_{y}"

                NUM_CLASSES = Base_Classes + x

                trainDataset, valDataset, testDataset, testTTDataset = getLoaders(dest_folder)

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

                log_dir = os.path.join(os.getcwd(), "logs", f"{Experiment_Name}", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
                
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

                epochs = EPOCHS
                with tf.device(USED_GPU):
                    history = model.fit(
                        trainDataset,
                        epochs=epochs,
                        validation_data=valDataset,
                        callbacks=[checkpoint_callback, tensorboard_callback],
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
                    metricsPath, f"trained_TT_Classes_{Experiment_Name}.json"
                )
                if os.path.exists(jsonPath):
                    os.remove(jsonPath)

                with open(jsonPath, "w") as f:
                    json.dump(saveDict, f, indent=2)
            except Exception as ex:
                error_message = f"An error occurred while experiment: {Experiment_Name}, running training: {ex}\n"
                print(error_message)
                log_error(error_message)
                
            try:
                tt_percentage(z, dest_folder)
            except Exception as ex:
                error_message = f"An error occurred while experiment: {Experiment_Name}, running tt_percentage {z}: {ex}\n"
                print(error_message)
                log_error(error_message)
        try:
            dir_delete(dest_folder)
        except Exception as ex:
            error_message = f"An error occurred while experiment: {Experiment_Name}, running dir_delete: {ex}\n"
            print(error_message)
            log_error(error_message)      
