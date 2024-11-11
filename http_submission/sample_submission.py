import numpy as np
# import pysaliency

class SampleScanpathModel():
    def __init__(self):
        super().__init__()

    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        return np.log(stimulus)