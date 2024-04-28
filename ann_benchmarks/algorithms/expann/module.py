import numpy as np
import expann_py as ep_nodim
import expann_py_64
import expann_py_128
import expann_py_256
import expann_py_768
import expann_py_832
import expann_py_1024

from ..base.module import BaseANN

class ExpAnnWrapper(BaseANN):
    def __init__(self, metric, index_param):
        self._m = index_param["M"]
        self._ef_construction = index_param["ef_construction"]
        self._ortho_count = index_param["ortho_count"]
        self._prune_overflow = index_param["prune_overflow"]
        self._use_compression = index_param["use_compression"]
        self.name = "expANN Anti-Topo Engine"
        self.res = None
        self.metric = metric
        self.modules = {
            64: expann_py_64,
            128: expann_py_128,
            256: expann_py_256,
            768: expann_py_768,
            832: expann_py_832,
            1024: expann_py_1024
        }

    def get_module_for_dim(self, d):
        for dim in sorted(self.modules.keys()):
            if d <= dim:
                return dim, self.modules[dim]
        return d, ep_nodim

    def fit(self, X):
        self.dim_unpadded = X.shape[1]
        self.dim_padded, self.epy = self.get_module_for_dim(self.dim_unpadded)
        print("God padded dim:", self.dim_padded)
        self.engine = self.epy.AntitopoEngine(self._m, self._ef_construction, self._ortho_count, self._prune_overflow, self._use_compression)
        for vector in X:
            #padded_vector = np.pad(vector, (0, self.dim_padded - self.dim_unpadded), 'constant')
            #v = self.epy.Vec(padded_vector.tolist())
            v = self.epy.Vec(vector)
            if self.metric == "angular":
                v.normalize()
            self.engine.store_vector(v)
        self.engine.build()

    def query(self, q, k):
        # padded_q = np.pad(q, (0, self.dim_padded - len(q)), 'constant')
        # q = self.epy.Vec(padded_q.tolist())
        q = self.epy.Vec(q)
        if self.metric == "angular":
            q.normalize()
        return self.engine.query_k(q, k)

    def set_query_arguments(self, ef_search):
        self.engine.set_ef_search(ef_search)
