import numpy as np
import pandas as pd

val_step = 0
en_test = pd.read_csv('/home/mlepekhin/data/en_train').sample(frac=1, random_state=42)

labels = np.unique(sorted(en_test.target.values))
label_to_id = {label: label_id for label_id, label in enumerate(labels)}
print(len(label_to_id), "LABELS")
    
dataset = []

for i in range(len(en_test)):
    if i >= val_step * 0.2 * len(en_test) and i < (val_step+1) * 0.2 * len(en_test):
        text = en_test.text.values[i]
        label = en_test.target.values[i]
        dataset.append((' '.join(text.split()[:300]), label_to_id[label]))