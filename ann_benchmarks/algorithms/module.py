import numpy as np
import expann_py

from ..base.module import BaseANN

print(expann_py)

class ExpAnnWrapper(BaseANN):
    # TODO what are the params to this?? args?
    def __init__(self, metric, index_param, query_param):
        # TODO: Actually use the params?
        self.engine = expann_py.expANN()
        self.name = self.engine.name()
        self.res = None
        self.metric = metric

    def fit(self, X):
        dim = X.shape[1]
        for vector in X:
            v = expann_py.Vec(vector.tolist())
            self.engine.store_vector(v)
        self.engine.build()

    def query(self, q, k):
        q = expann_py.Vec(q.to_list())
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

    def set_query_arguments(self, query_args):
        # TODO: Actually do this?
        pass

    def load_index(self, dataset):
        return False

    def get_results(self):
        if self.res is None:
            raise ValueError("Run a query before getting results")
        return self.res

    def __str__(self):
        return self.name
