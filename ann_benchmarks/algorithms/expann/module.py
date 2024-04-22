import numpy as np
import expann_py

from ..base.module import BaseANN

print(expann_py)

class ExpAnnWrapper(BaseANN):
    def __init__(self, metric, index_param):
        self._m = index_param["M"]
        self._ef_construction = index_param["ef_construction"]
        self._ortho_count = index_param["ortho_count"]
        self._ortho_factor = index_param["ortho_factor"]
        self._ortho_bias = index_param["ortho_bias"]
        self._prune_overflow = index_param["prune_overflow"]
        self.engine = expann_py.AntitopoEngine(self._m, self._ef_construction, self._ortho_count, self._ortho_factor, self._ortho_bias, self._prune_overflow)
        self.name = self.engine.name()
        self.res = None
        self.metric = metric

    def fit(self, X):
        dim = X.shape[1]
        for vector in X:
            v = expann_py.Vec(vector.tolist())
            if self.metric == "angular":
                v.normalize()
            self.engine.store_vector(v)
        self.engine.build()

    def query(self, q, k):
        q = expann_py.Vec(q.to_list())
        if self.metric == "angular":
            q.normalize()
        return self.engine.query_k(q, k)
        #query_vectors = []
        #for query_vector in Q:
        #    q = expann_py.Vec(query_vector.tolist())
        #    query_vectors.append(q)
        #result_indices = self.engine.query_k_batch(query_vectors, k)

        #max_len = len(max(result_indices, key=len))
        #for result_index in result_indices:
        #    while len(result_index) < max_len:
        #        result_index.append(0)
        #self.res = np.array(result_indices)

    def set_query_arguments(self, ef_search):
        self.engine.set_ef_search(ef_search)
