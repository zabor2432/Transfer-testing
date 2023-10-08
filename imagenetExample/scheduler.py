import os
import numpy as np

for setPortion in np.arange(0.05,1.05,0.05):
    print(f"processing subset with {round(setPortion,2)}% of test class")
    _ = os.popen(f"python3 smallDataTest.py --testedDataShare {round(setPortion,2)}").read()
