import json
import numpy as np
from tqdm import tqdm

with open("labrec.json", 'r') as f:
    recording = json.load(f)
for ts in tqdm(recording):
    arr = np.array(recording[ts])
    arr *= [180/np.pi, 1000]
    # arr = np.sort(arr, 0)
    arr = np.concatenate([[[15]]*len(arr), arr], 1)
    # print(arr)
    # exit()
    recording[ts] = {
        "quality": arr[:,0].tolist(),
        "degrees": arr[:,1].tolist(),
        "milimeters": arr[:,2].tolist()
    }
with open("labrecfix.json", 'w') as f:
    json.dump(recording, f, indent=2)