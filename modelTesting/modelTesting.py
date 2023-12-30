import os
import tensorflow as tf 
from keras.models import load_model
import json
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 32
MODELS_PATH = '/home/macierz/s181655/projektBadawczy/Transfer-testing/Books/models'
TEST_PATH = '/home/macierz/s181655/projektBadawczy/Transfer-testing/Books/BookDatasetC_D'
METRICS_PATH = '/home/macierz/s177788/projektBadawczy/Transfer-testing/modelTesting'

def getLoaders(dbPath: str):
    testDir = os.path.join(dbPath, "test")
    testTTDir = os.path.join(dbPath, "test_T")

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

    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    testTT_dataset = testTT_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return test_dataset, testTT_dataset

if __name__ == '__main__':

    test_dataset, testTT_dataset = getLoaders(TEST_PATH)
    
    for modelCurr, _ , avFiles in os.walk(MODELS_PATH):
        avFiles.sort()
        if len(avFiles) == 0:
            continue
    
        dicName = modelCurr.split(os.sep)[-1]

        if "CD_1_" not in dicName:
            continue

        metDic = {}
        testLossArr=[]
        testAccuracyArr =[]
        testTTLossArr=[]
        testTTAccuracyArr =[]
        epochs = np.arange(1, len(avFiles)+1, 1)
        
        for file in avFiles:
            if not '.h5' in file:
                continue
            model = load_model(os.path.join(modelCurr, file))
            metrics = model.evaluate(test_dataset)
            metricsT = model.evaluate(testTT_dataset)

            metDic[file] = {"testLoss": metrics[0],
                            "testAccuracy": metrics[1],
                            "testTLoss": metricsT[0],
                            "testTAccuracy": metricsT[1]}
            testLossArr.append(metrics[0])
            testAccuracyArr.append(metrics[1])
            testTTLossArr.append(metricsT[0])
            testTTAccuracyArr.append(metricsT[1])
        plt.plot(epochs, testLossArr, color = "deeppink", label = "Test Loss")
        plt.plot(epochs, testTTLossArr, label = "TestTT Loss")
        plt.legend()
        plt.title(dicName)
        plt.xlabel("Epochs")
        plt.xticks(np.arange(1, len(avFiles)+1, 1))
        plt.ylabel("Loss")
        plt.savefig(os.path.join(METRICS_PATH, f'{dicName}_LOSS.png'))
        plt.close()

        plt.plot(epochs, testAccuracyArr, color = "deeppink", label = "Test Accuracy")
        plt.plot(epochs, testTTAccuracyArr, label = "TestTT Accuracy")
        plt.legend()
        plt.title(dicName)
        plt.xlabel("Epochs")
        plt.xticks(np.arange(1, len(avFiles)+1, 1))
        plt.ylabel("Accuracy")
        plt.savefig(os.path.join(METRICS_PATH, f'{dicName}_ACC.png'))
        plt.close()

            

        with open(os.path.join(METRICS_PATH, f'{dicName}.json'), "w") as f:
            json.dump(metDic, f, indent=2)




            # predykcja
            # porównanie z gt
            #policz metryki
            # dodaj metryki do metDic
        # zapisz metDic to pliku .json

