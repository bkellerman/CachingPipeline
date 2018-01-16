import os
import hashlib
import numpy as np
import cache

class CachingPipeline(object):
    """ Wrapper for sklearn.pipeline.Pipeline that caches all tranformed data. """
    def __init__(self, pipeline, cache_dir, verbose=0, use_cache=True):
        """ Docstring """
        self.pipeline = pipeline
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.use_cache = use_cache

        self.steps = pipeline.steps
        self.named_steps = pipeline.named_steps
        self.final_est = self.steps[-1][-1]
        self.pipe_params = str(sorted(self.pipeline.get_params()))

    def get_params(self, deep=True):
        """ Docstring """
        return self.pipeline.get_params(deep)

        """
        #Trying to get RandomSearchCV to work
        pipe_params = self.pipeline.get_params(deep)
        pipe_params.update({'pipeline': self.pipeline, 'cache_dir': self.cache_dir})
        return pipe_params
        """

    def set_params(self, **kwargs):
        """ Set params of the pipeline"""
        self.pipeline.set_params(**kwargs)

    def est_fit_params(self, fit_params):
        """ Return final estimator fit_params from pipeline fit params. """
        est_name = self.pipeline.steps[-1][0]
        est_fit_params = {}

        for k, v in fit_params.items():
            tf, value = k.split('__')
            if tf == est_name:
                est_fit_params[value] = v

        return est_fit_params

    def to_json(self):
        """ Keras specific """
        return self.pipeline.steps[-1][-1].model.to_json()

    def array_hash(self, X):
        """ Create a unique hash of a numpy array."""
        data_hash = cache.subsample_hash(X)
        hash_params = self.pipe_params + data_hash
        h = hashlib.sha256()
        h.update(hash_params.encode('utf-8'))
        return h.hexdigest()

    def save_cache(self, X, X_tf):
        """ Save a tranformed numpy array """
        if self.verbose:
            print("%s begin save_cache. X shape %s" % (self.__class__.__name__, X.shape))

        cache_hash = self.array_hash(X)
        if self.verbose:
            print("Attempting to save pipeline transformed array with cache hash %s" % cache_hash)

        X_file = '.'.join(["X", cache_hash, "npy"])

        try:
            np.save(os.path.join(self.cache_dir, X_file), X_tf)

        except Exception as e:
            print(e)
            raise ValueError("Pipeline data could not be saved.")

    def get_cache(self, X):
        """ Retrieve a transformed numpy array from cache """
        if self.verbose:
            print("%s begin cached. X shape %s." % (self.__class__.__name__, X.shape))

        cache_hash = self.array_hash(X)

        if self.verbose:
            print("Attempting to load pipeline transformed array with cache hash %s" % cache_hash)

        X_file = '.'.join(["X", cache_hash, "npy"])

        try:
            X_cached = np.load(os.path.join(self.cache_dir, X_file))
            if self.verbose:
                print("Successfully loaded cached pipeline arrays. %s" % (str(X.shape)))
            return X_cached

        except Exception:
            if self.verbose:
                print("Couldn't load cached pipeline arrays. %s" % (str(X.shape)))
            return None

    def est_fit(self, X, y, **kwargs):
        """ Fit final estimator of the pipeline """
        if self.verbose:
            print("%s begin est_fit. X shape %s. y shape %s" %
                  (self.__class__.__name__, X.shape, y.shape))

        fit_params = self.est_fit_params(kwargs)

        try:
            return self.pipeline.steps[-1][-1].fit(X, y, **fit_params)
        except Exception as e:
            print(e)
            raise ValueError("Final estimator fit failed.")

    def predict_proba(self, X):
        """ Predict_proba X against final estimator """
        if self.verbose:
            print("%s begin predict_proba. X shape %s" % (self.__class__.__name__, X.shape))
        if not self.use_cache:
            return self.pipeline.predict_proba(X)

        X_tf = self.get_cache(X)

        if X_tf is None:
            X_tf = self.tf_transform(X)
            self.save_cache(X, X_tf)

        return self.pipeline.steps[-1][-1].predict_proba(X_tf)

    def predict(self, X):
        """ Predict X against final estimator """
        if self.verbose:
            print("%s begin predict_proba. X shape %s" % (self.__class__.__name__, X.shape))
        if not self.use_cache:
            return self.pipeline.predict(X)

        X_tf = self.get_cache(X)

        if X_tf is None:
            X_tf = self.tf_transform(X)
            self.save_cache(X, X_tf)

        return self.pipeline.steps[-1][-1].predict(X_tf)

    def fit(self, X, y, **kwargs):
        """ Fit entire pipeline """
        if self.verbose:
            print("%s begin fit. X shape %s. y shape %s" %
                  (self.__class__.__name__, X.shape, y.shape))
        if not self.use_cache:
            self.pipeline.fit(X, y, **kwargs)
            return self

        X_tf = self.get_cache(X)
        if X_tf is None:
            if self.verbose:
                print("Cached returned none")
            self.tf_fit(X, y)
            X_tf = self.tf_transform(X)
            self.save_cache(X, X_tf)
            self.est_fit(X_tf, y, **kwargs)
        else:
            fit_params = self.est_fit_params(kwargs)
            self.est_fit(X_tf, y, **kwargs)
            #self.pipeline.steps[-1][-1].fit(X_tf, y, **fit_params)

        return self

    def tf_transform(self, X):
        """ Transform X with all tranformers of the pipeline"""
        if self.verbose:
            print("%s begin tf_transform. X shape %s." % (self.__class__.__name__, X.shape))

        X_tf = self.get_cache(X)

        if X_tf is not None:
            return X_tf

        clf_name, clf = self.pipeline.steps.pop()
        X_tf = self.pipeline.transform(X)
        self.save_cache(X, X_tf)
        self.pipeline.steps.append((clf_name, clf))

        return X_tf

    def tf_fit(self, X, y):
        """ Fit all tranformers of the pipeline"""
        if self.verbose:
            print("%s begin tf_fit. X shape %s. y shape %s" %
                  (self.__class__.__name__, X.shape, y.shape))

        X_tf = self.get_cache(X)

        if X_tf is not None:
            return

        clf_name, clf = self.pipeline.steps.pop()
        X_t = self.pipeline.fit_transform(X, y)
        self.save_cache(X, X_t)
        self.pipeline.steps.append((clf_name, clf))
