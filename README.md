# SKCachingPipeline

The module wraps `sklearn.pipeline.Pipeline` and provides caching of transformed numpy arrays.  It is different from the `memory` parameter of `Pipeline`, which only caches the fitted tranformers.  When a `Pipeline` has transformers with computationally expensive transforms(), this module can significantly reduce total `Pipeline` `fit()` time after the initial transform has been cached.

`CachingPipeline` only caches the final output of all transformers in the `Pipeline`.   It will not cache tranformed data for each transformer.

The transformed array is saved using a hash of the `Pipeline` params and numpy array input.  If any of the transformer params or numpy input changes, a new cache will be used/checked.

Example
-------
 
    
```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.decomposition import PCA 
from sklearn.datasets import make_classification
from pipe import CachingPipeline

CACHE_DIR = 'cache.tmp'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

X, y = make_classification(100000, 50, random_state=43)

cl = LogisticRegression()

pipeline = Pipeline([
                     ('im', Imputer()),
                     ('ss', StandardScaler()),
                     ('pca', PCA(20, whiten=True)),
                     ('cl', cl)])

cp = CachingPipeline(pipeline, CACHE_DIR)

start = time.time()
cp.fit(X, y)
print("Initial Uncached Pipeline.fit() time: %0.1f" % (time.time() - start))

start = time.time()
for _ in range(10):
    cp.fit(X, y)
print("10 Cached Pipeline.fit()'s time: %0.1f" % (time.time() - start))


Output:

Initial Uncached Pipeline.fit() time: 1.5
10 Cached Pipeline.fit()'s time: 3.1
```






