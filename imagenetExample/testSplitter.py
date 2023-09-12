import os
import shutil
import random

# sorry for the hardcoded parts, I'm in a rush rn (also it might be buggy)

# sample randomly 25 images from test dir and move rest to tempDir to be used in tests

TEST_CLASS_DIR = "/home/macierz/s177788/projektBadawczy/data-small/test/n04579145"

sampleCount = len(os.listdir(TEST_CLASS_DIR))

os.makedirs("/home/macierz/s177788/projektBadawczy/data-small/tempDir", exist_ok=True)
os.makedirs(os.path.join("/home/macierz/s177788/projektBadawczy/data-small", "val", TEST_CLASS_DIR), exist_ok=True)

for i in range(sampleCount-25):
    sample = random.choice(os.listdir(TEST_CLASS_DIR))
    shutil.move(os.path.join(TEST_CLASS_DIR,sample), "/home/macierz/s177788/projektBadawczy/data-small/tempDir")

for i in range(5):
    sample = random.choice(os.listdir(TEST_CLASS_DIR))
    shutil.move(os.path.join(TEST_CLASS_DIR,sample), os.path.join("/home/macierz/s177788/projektBadawczy/data-small", "val", TEST_CLASS_DIR))

for className in  os.listdir("/home/macierz/s177788/projektBadawczy/data-small/train"):
    os.makedirs(os.path.join("/home/macierz/s177788/projektBadawczy/data-small/test", className), exist_ok=True)