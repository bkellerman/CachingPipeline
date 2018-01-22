# SKCachingPipeline

The module wraps sklearn.pipeline.Pipeline and provides caching of transformed numpy arrays.  It is different from the 'memory' parameter of sklearn's Pipeline, which only caches the fitted tranformers.  When your Pipeline has Transformers with computationally expensive transforms(), this module can significantly reduce total Pipeline fit() time after the initial transform has been cached.

CachingPipeline only caches the final output of all Transformers in the Pipeline.   It will not cache tranformed data for each Transformer.


The transformed array is saved using a hash of the Pipeline params and numpy array input.  If any of the Transformer params or numpy input changes, a new cache will be used/checked.

TODO:
    Make it work with Sklearn's GridSearchCV and RandomizedSearchCV 

