import os
import numpy as np
import hashlib

def subsample_hash(a):
    """ Returns a unique hash for a numpy array by sampling values """
    rng = np.random.RandomState(113)
    inds = rng.randint(low=0, high=a.size, size=1000)
    b = a.flat[inds]
    m = hashlib.sha256()
    s = b.tostring()
    m.update(b.tostring())

    return m.hexdigest()
