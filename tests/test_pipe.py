from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Imputer
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import PCA 
from sklearn.datasets import make_classification
from pipe import CachingPipeline

import numpy as np
import time

X = np.array([[[1.,2,3,4,5],[4,5,6,7,8],[1,2,3,4,5], [2,4,5,6,7]],
              [[2,3,4,5,6],[4,5,6,7,8],[2,3,4,5,6], [4,5,6,7,8]],
              [[2,3,4,5,6],[4,5,6,7,8],[2,3,4,5,6], [2,3,4,5,6]]])

X, y = make_classification(100000, 50) 

cl = RandomForestClassifier(3)

pipeline = Pipeline([
                     ('im', Imputer()),
                     ('ss', StandardScaler()),
                     ('pca', PCA(20, whiten=True)),
                     ('cl', cl)])

cp = CachingPipeline(pipeline, 'cache_tmp')

start = time.time()
cp.fit(X, y)
print("Single Uncached Pipeline.fit() time: %0.1f" % (time.time() - start))

start = time.time()
for _ in range(5):
    cp.fit(X, y)
print("5 Cached Pipeline.fit()'s time: %0.1f" % (time.time() - start))


