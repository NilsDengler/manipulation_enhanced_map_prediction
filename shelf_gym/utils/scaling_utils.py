import numpy as np

SCALE_T = 1.8

def scale_semantic_probs(semantic_probs,t = SCALE_T):
    for i in range(len(semantic_probs)):
        tmp = semantic_probs[i]
        tmp = np.power(tmp,t)
        tmp = tmp/tmp.sum(axis =-1,keepdims = True)
        semantic_probs[i] = tmp
    return semantic_probs

def scale_semantic_map(semantic_map,t = SCALE_T,axis = -1):
    semantic_map = np.power(semantic_map,t)
    semantic_map = semantic_map/semantic_map.sum(axis = axis,keepdims = True)
    return semantic_map