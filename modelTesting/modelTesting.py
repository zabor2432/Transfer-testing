import os
import tensorflow as tf 
from keras.models import load_model
import json
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 32
MODELS_PATH = '/home/macierz/s181655/projektBadawczy/Transfer-testing/Books/models'
TEST_PATH = '/home/macierz/s181655/projektBadawczy/Transfer-testing/Books'
METRICS_PATH = '/home/macierz/s177788/projektBadawczy/Transfer-testing/modelTesting'
CONSIDERED_MODELS = ['AB_1_10', 'AB_1_25', 'AB_1_50', 'AB_1_100', 'AB_3_10', 'AB_3_25', 'AB_3_50', 'AB_3_100', 'AB_10_10', 'AB_10_25', 'AB_10_50', 'AB_10_100', 'AB_30_10', 'AB_30_25', 'AB_30_50', 
                     'AB_30_100',  'AD_3_10', 'AD_3_25', 'AD_3_50', 'AD_3_100', 'CD_1_10', 'CD_1_25', 'CD_1_50', 'CD_1_100', 'CD_3_10', 'CD_3_25', 'CD_3_50', 'CD_3_100', 'CD_10_10', 'CD_10_25', 
                     'CD_10_50', 'CD_10_100', 'GH_1_100', 'GH_3_100','CD_30_10', 'CD_30_25', 'CD_3_50', 'CD_30_100', 'CD_1_50', 'CD_1_100']                 

def getLoaders(dbPath: str):
    testDir = os.path.join(dbPath, "test")
    testTDir = os.path.join(dbPath, "test_T")
    testTTDir = os.path.join(dbPath, "test_TT")

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        testDir,
        labels="inferred",
        label_mode="categorical",
        batch_size=BATCH_SIZE,
        image_size=(224, 224),
        shuffle=False,
    )

    testT_dataset = tf.keras.utils.image_dataset_from_directory(
        testTDir,
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
    testT_dataset = testT_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)    
    testTT_dataset = testTT_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return test_dataset, testT_dataset, testTT_dataset

if __name__ == '__main__':

    filesInBooks = os.listdir(TEST_PATH)
    modelsList = os.listdir(MODELS_PATH)
    filteredFiles = [x for x in filesInBooks if x.startswith('TestDataset')]

    for testset in filteredFiles:
        testName = testset.replace('TestDataset', '')
        testName =  testName[:1] + testName[2:] + '_'

        currModels = [x for x in CONSIDERED_MODELS if x.startswith(testName)]
        bigDataVersions = [testset.replace("TestDataset", "Big_Data_") + x.replace(testName, "_") for x in currModels]
        bigDataVersions = [x for x in bigDataVersions if x in modelsList]

        currModels = currModels + bigDataVersions

        test_dataset,testT_dataset,testTT_dataset = getLoaders(os.path.join(TEST_PATH, testset))
        
        for modelName in currModels: 
            if "Big_Data" not in modelName: # comment if everything without BigData
                continue

            modelCurr = os.path.join(MODELS_PATH, modelName)
            avFiles = os.listdir(modelCurr)

            metDic = {}
            testLossArr=[]
            testAccuracyArr =[]
            testTLossArr=[]
            testTAccuracyArr =[]            
            testTTLossArr=[]
            testTTAccuracyArr =[]
            epochs = np.arange(1, len(avFiles)+1, 1)
            
            for file in avFiles:
                if not '.h5' in file:
                    continue
                model = load_model(os.path.join(modelCurr, file))
                metrics = model.evaluate(test_dataset)
                metricsT = model.evaluate(testT_dataset)
                metricsTT = model.evaluate(testTT_dataset)                

                metDic[file] = {"testLoss": metrics[0],
                                "testAccuracy": metrics[1],
                                "testTLoss": metricsT[0],
                                "testTAccuracy": metricsT[1],
                                "testTTLoss": metricsTT[0],
                                "testTTAccuracy": metricsTT[1]}
                
                testLossArr.append(metrics[0])
                testAccuracyArr.append(metrics[1])
                testTLossArr.append(metricsT[0])
                testTAccuracyArr.append(metricsT[1])
                testTTLossArr.append(metricsTT[0])
                testTTAccuracyArr.append(metricsTT[1])


            plt.plot(epochs, testLossArr, color = "deeppink", label = "Test Loss")
            plt.plot(epochs, testTLossArr, color = "darkgreen", label = "TestT Loss")            
            plt.plot(epochs, testTTLossArr, label = "TestTT Loss")
            plt.legend()
            plt.title(modelName)
            plt.xlabel("Epochs")
            plt.xticks(np.arange(1, len(avFiles)+1, 1))
            plt.ylabel("Loss")
            plt.savefig(os.path.join(METRICS_PATH, f'{modelName}_LOSS.png'))
            plt.close()

            plt.plot(epochs, testAccuracyArr, color = "deeppink", label = "Test Accuracy")
            plt.plot(epochs, testTAccuracyArr, color = "darkgreen", label = "TestT Accuracy")              
            plt.plot(epochs, testTTAccuracyArr, label = "TestTT Accuracy")
            plt.legend()
            plt.title(modelName)
            plt.xlabel("Epochs")
            plt.xticks(np.arange(1, len(avFiles)+1, 1))
            plt.ylabel("Accuracy")
            plt.savefig(os.path.join(METRICS_PATH, f'{modelName}_ACC.png'))
            plt.close()
                

            with open(os.path.join(METRICS_PATH, f'{modelName}.json'), "w") as f:
                json.dump(metDic, f, indent=2)

