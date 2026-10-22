# src/utils/DeepAbstractMethod.py
import numpy as np

def find_channel_medoid(fps, members):
    if len(members) == 1:
        return members[0]
    sub_fps = fps[members]
    centroid = sub_fps.mean(axis=0)
    dists = np.array([
        1.0 - np.dot(fp, centroid) / (np.linalg.norm(fp) * np.linalg.norm(centroid) + 1e-8)
        for fp in sub_fps
    ])
    return members[np.argmin(dists)]