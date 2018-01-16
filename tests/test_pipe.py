import sys
from tech import IndicatorScaler, BBandFeature, MADiffFeature, IdentityFeature, SeqTransformer, TwoDTransformer
from tech import PCATransformer, CompositeFeature, CurrentScaler, StochFeature, RSIFeature, MACDFeature
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from pipe import CachedPipeline, CachedPipeline2
import numpy as np
import time
import cache
#np.random.seed(1)
back_size = 50
current_size = 30
NUM_PCA = 20

X = np.array([[[1.,2,3,4,5],[4,5,6,7,8],[1,2,3,4,5], [2,4,5,6,7]],
              [[2,3,4,5,6],[4,5,6,7,8],[2,3,4,5,6], [4,5,6,7,8]],
              [[2,3,4,5,6],[4,5,6,7,8],[2,3,4,5,6], [2,3,4,5,6]]])
X = np.random.random((50000, 4, back_size + current_size))
print(X.shape)
X[:, :4,:] += 20

cl = RandomForestClassifier()
seq = SeqTransformer(back_size, with_prices=True, close_only=True)

pipeline = Pipeline([
                     ('rsi14', RSIFeature(14)),
                     ('st14', StochFeature(14, 3, 0, 3, 0)),
                     ('macd26', MACDFeature(12, 26, 9)),
                     ('c1', CurrentScaler(back_size)),
                     ('two', TwoDTransformer(back_size, with_prices=True, close_only=True, export=False)),
                     ('ss', StandardScaler()),
                     ('cl', cl)])

pipeline2 = Pipeline([
                     ('rsi14', RSIFeature(14)),
                     ('st14', StochFeature(14, 3, 0, 3, 0)),
                     ('macd26', MACDFeature(12, 26, 9)),
                     ('c1', CurrentScaler(back_size)),
                     ('two', TwoDTransformer(back_size, with_prices=True, close_only=True, export=False)),
                     ('ss', StandardScaler()),
                     ('cl', cl)])

pipeline3 = Pipeline([
                     ('rsi14', RSIFeature(14)),
                     ('two', TwoDTransformer(back_size, with_prices=True, close_only=True, export=False)),
                     ('cl', cl)])

pipeline4 = Pipeline([
                     ('rsi14', RSIFeature(14)),
                     ('two', TwoDTransformer(back_size, with_prices=True, close_only=True, export=False)),
                     ('cl', cl)])

cp = CachedPipeline(pipeline, 'pipe_cache', verbose=0)
cp2 = CachedPipeline2(pipeline2, 'pipe_cache', verbose=0)
y = np.zeros(X.shape[0])

start = time.time()
cp.fit(X, y)
cp2.fit(X, y)
print("Uncached Pipeline fit time: %0.1f" % (time.time() - start))
start = time.time()
for _ in range(10):
    cp.fit(X, y)
    cp2.fit(X, y)
print("10 Cached Pipeline fits time: %0.1f" % (time.time() - start))

start = time.time()
cp.pipeline.steps.pop()
X_t = cp.transform(X)
X_t2 = cp2.transform(X)
print("Uncached Pipeline transform time: %0.1f" % (time.time() - start))
start = time.time()
for _ in range(10):
    #cp.pipeline.steps.pop()
    X_t = cp.transform(X)
    X_t2 = cp2.transform(X)
print("10 Cached Pipeline tranforms time: %0.1f" % (time.time() - start))
assert(np.allclose(X_t, X_t2, atol=1e-07, equal_nan=True))

"""
assert(np.allclose(X_comp_C, X_1_C, atol=1e-04, equal_nan=True))
assert(np.allclose(X_comp_F, X_1_F, atol=1e-04, equal_nan=True))
assert(np.allclose(X_comp_C, X_1_F, atol=1e-04, equal_nan=True))
"""
