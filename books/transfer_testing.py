import os
import random
import numpy as np
import shutil
import json
import datetime
from argparse import ArgumentParser


# HOW TO USE THE SCRIPT:
# python3 transfer_testing.py --Used_GPU <> --Base_Class <> --Num_Base <> --Added_Class <> --Num_Added <>

# <> -> insert a value for example "--Base_Class A"

#Here add paths to main and transfer data:

book_paths = {
    'A': '/local_storage/gwo/public/transfer-testing/books/CB/978-83-7420-600-6',
    'B': '/local_storage/gwo/public/transfer-testing/books/CB/978-83-7420-629-7'
}

#DATA LABELS HAVE TO BE IN NUMERICAL FORMAT: 000,001,002.....

#Specify main path you run experiments
mainPath = f'/home/macierz/s181655/projektBadawczy/Transfer-testing/Books/'


BATCH_SIZE = 32
INPUT_SHAPE = (224, 224, 3)
EPOCHS = 10


def getLoaders(dbPath: str):
    trainDir = os.path.join(dbPath, "train")
    valDir = os.path.join(dbPath, "val")
    testDir = os.path.join(dbPath, "test")

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

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset


def preprocess(images, labels):
    return preprocess_input(images), labels

def testPrepare(destination, Num_Base):
    # Create empty directories "000" to "0xx" for test set
    for i in range(Num_Base):
        directory_name = str(i).zfill(3)
        directory_path = os.path.join(f'{mainPath}{destination}/test', directory_name)
        
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)

def book_prepare(num_folders, destination, added_class):    

    num_folders_to_copy = num_folders

    #Adjust the path
    d= f'{mainPath}{destination}'

    #Add TT classes to train set
    #Following script loads first n pages (classes) of a book.
    for i, class_folder in enumerate(sorted(os.listdir(os.path.join(added_class, 'train')))):
        if i >= 0:  
            if i < num_folders_to_copy:  # Limit the number of folders to copy
            #if i - 2 < num_folders_to_copy:  # Limit the number of folders to copy # in this case following script loads n pages (classes) of a book starting from page 3.
                source_class_path = os.path.join(added_class, 'train', class_folder)
                target_class_folder = class_folder + '_TT'
                target_class_path = os.path.join(d, 'train', target_class_folder)

                shutil.copytree(source_class_path, target_class_path)
            else:
                break

    #Add TT classes to val set
    for i, class_folder in enumerate(sorted(os.listdir(os.path.join(added_class, 'val')))):
        if i >= 0:  
            if i < num_folders_to_copy:  # Limit the number of folders to copy
            #if i - 2 < num_folders_to_copy:  # Limit the number of folders to copy # in this case following script loads n pages (classes) of a book starting from page 3.
                source_class_path = os.path.join(added_class, 'val', class_folder)
                target_class_folder = class_folder + '_TT'
                target_class_path = os.path.join(d, 'val', target_class_folder)

                shutil.copytree(source_class_path, target_class_path)
            else:
                break
    #Add TT classes to test set
    for i, class_folder in enumerate(sorted(os.listdir(os.path.join(added_class, 'test')))):
        if i >= 0:  
            if i  < num_folders_to_copy:  # Limit the number of folders to copy
            #if i - 2 < num_folders_to_copy:  # Limit the number of folders to copy # in this case following script loads n pages (classes) of a book starting from page 3.
                source_class_path = os.path.join(added_class, 'test', class_folder)
                target_class_folder = class_folder + '_TT'
                target_class_path = os.path.join(d, 'test', target_class_folder)

                shutil.copytree(source_class_path, target_class_path)
            else:
                break

def log_error(error_msg):
    from datetime import datetime
    error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #Adjust the path
    log_filename = f"{mainPath}Errors/error_log_{error_time}.txt"
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
        "--Num_Base",
        dest="Num_Base",
        type=int,
        help="set number of base classes",
    ) 
    
    parser.add_argument(
        "--Added_Class",
        dest="Added_Class",
        type=str,
        help="choose added (TT) class",
    )
    
    parser.add_argument(
        "--Num_Added",
        dest="Num_Added",
        type=int,
        help="set number of added (TT) classes",
    )  
    
    args = parser.parse_args()
    
    USED_GPU = f"GPU:{args.Used_GPU}"
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.Used_GPU}"
    
    base_class = book_paths[args.Base_Class]
    Num_Base = args.Num_Base
    added_class = book_paths[args.Added_Class]
    Num_Added = args.Num_Added
    dest_folder = f'BookDataset{args.Base_Class}_{args.Added_Class}'
    experiment_prefix = f"{args.Base_Class}{args.Added_Class}"


    import tensorflow as tf
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.layers import (
        Dense,
        Flatten,
    )
    from tensorflow.keras.models import Sequential
    from tensorflow.python.client import device_lib
    

    
    try:
        book_prepare(Num_Added, dest_folder, added_class)
    except Exception as ex:
        error_message = f"An error occurred in experiment: {experiment_prefix}_{Num_Added} while book_prepare {Num_Added}: {ex}\n"
        print(error_message)
        log_error(error_message)
            
    try:
        testPrepare(dest_folder, Num_Base)
    except Exception as ex:
        error_message = f"An error occurred in experiment: {experiment_prefix}_{Num_Added} while testPrepare {Num_Added}: {ex}\n"
        print(error_message)
        log_error(error_message)     
    
    try:
        random.seed(123)

        Experiment_Name = f"{experiment_prefix}_{Num_Added}"

        NUM_CLASSES = Num_Base + Num_Added

        trainDataset, valDataset, testDataset = getLoaders(dest_folder)

        base_model = VGG16(
            weights="imagenet", include_top=False, input_shape=INPUT_SHAPE
        )
        base_model.trainable = False 

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

        
        with tf.device(USED_GPU):
            history = model.fit(
                trainDataset,
                epochs=EPOCHS,
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

        metricsPath = os.path.join(os.getcwd(), "results", Experiment_Name)
        os.makedirs(metricsPath, exist_ok=True)

        saveDict = {
            "train_accuracy": round(train_accuracy[best_epoch], 3),
            "val_accuracy": round(val_accuracy[best_epoch], 3),
            "test_accuracy": round(test_accuracy, 3),
            "test_loss": round(test_loss, 3),
        }

        jsonPath = os.path.join(
            metricsPath, f"TT_{Experiment_Name}.json"
        )
        if os.path.exists(jsonPath):
            os.remove(jsonPath)

        with open(jsonPath, "w") as f:
            json.dump(saveDict, f, indent=2)
    except Exception as ex:
        error_message = f"An error occurred while experiment: {Experiment_Name}, running training: {ex}\n"
        print(error_message)
        log_error(error_message)
                

 
