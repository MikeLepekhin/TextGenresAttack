import numpy as np
import pandas as pd

val_step = 0
ru_train = pd.read_csv('/home/mlepekhin/data/ru_train').sample(frac=1, random_state=42)

labels = np.unique(sorted(ru_train.target.values))
label_to_id = {label: label_id for label_id, label in enumerate(labels)}
print(len(label_to_id), "LABELS")
    
dataset = []

for i in range(len(ru_train)):
    if i >= val_step * 0.2 * len(ru_train) and i < (val_step+1) * 0.2 * len(ru_train):
        text = ru_train.text.values[i]
        label = ru_train.target.values[i]
        dataset.append((' '.join(text.split()[:300]), label_to_id[label]))