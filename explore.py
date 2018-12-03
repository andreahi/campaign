from collections import Counter

import numpy as np
import pandas as pd

from FileUtils import load_obj

df = pd.read_json("train.txt")




def ascii_histogram(seq) -> None:
    """A horizontal frequency-table/histogram plot."""
    counted = Counter(seq)
    for k in sorted(counted):
        print('{0:5d} {1}'.format(k, '+' * counted[k]))


target_indexs = load_obj("target_indexs")
print(target_indexs)


correct = np.argmax(np.array(list(np.array(df.newProducts))), axis=1)

ascii_histogram(correct)
