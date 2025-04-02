# Measures of Central Tendency

import numpy as np
import pandas as pd
from scipy import stats

data = [20, 30, 40, 50, 60, 70, 80, 90, 100]

# Mean
mean = np.mean(data)
print("Mean:", mean)

# Median
median = np.median(data)
print("Median:", median)

# Mode
mode_result = stats.mode(data, keepdims=False)
if len(mode_result.mode) > 0:  # Check if a mode exists
    print("Mode:", mode_result.mode[0], "Count:", mode_result.count[0])
else:
    print("No mode found")