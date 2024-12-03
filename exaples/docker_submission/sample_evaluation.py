import numpy as np
import sys
from sample_submission import MySimpleScanpathModel
from pysaliency.http_models import HTTPScanpathModel
sys.path.insert(0, '..')
import pysaliency


if __name__ == "__main__":
    http_model = HTTPScanpathModel("http://localhost:4000")
    http_model.check_type()

    # for testing
    model = MySimpleScanpathModel()

    # get MIT1003 dataset
    stimuli, fixations = pysaliency.get_mit1003(location='pysaliency_datasets')

    eval_fixations = fixations[fixations.scanpath_history_length > 0]

    for fixation_index in range(10):
        # get server response for one stimulus
        server_density = http_model.conditional_log_density(
            stimulus=stimuli.stimuli[eval_fixations.n[fixation_index]], 
            x_hist=eval_fixations.x_hist[fixation_index], 
            y_hist=eval_fixations.y_hist[fixation_index], 
            t_hist=eval_fixations.t_hist[fixation_index]
        )
        # get model response
        model_density = model.conditional_log_density(
            stimulus=stimuli.stimuli[eval_fixations.n[fixation_index]], 
            x_hist=eval_fixations.x_hist[fixation_index], 
            y_hist=eval_fixations.y_hist[fixation_index], 
            t_hist=eval_fixations.t_hist[fixation_index]   
        )

        # Testing 
        test = np.testing.assert_allclose(server_density, model_density)
        