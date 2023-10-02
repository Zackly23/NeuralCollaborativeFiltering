import torch
import pandas as pd
import numpy as np

data = {'no': np.arange(0,10),
        'nama': ['dav'+str(i) for i in range(10)],
        'nim': np.random.randn(10)}

df = pd.DataFrame(data)
sample = df.filter(['nama', 'nim']).value_counts()
print(sample)