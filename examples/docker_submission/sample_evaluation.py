import numpy as np
import sys
from sample_submission import MySimpleScanpathModel
from pysaliency.http_models import HTTPScanpathModel
sys.path.insert(0, '..')
import pysaliency


from tqdm import tqdm

import deepgaze_pytorch


if __name__ == "__main__":

    # initialize HTTPScanpathModel
    http_model = HTTPScanpathModel("http://localhost:4000")
    http_model.check_type()

    # for testing
    test_model = deepgaze_pytorch.DeepGazeIII(pretrained=True)

    # get MIT1003 dataset
    stimuli, fixations = pysaliency.get_mit1003(location='pysaliency_datasets')
    for stimulus in tqdm(stimuli):
        stimulus.stimulus_data

    # filter for scanpaths with 3 < fixations for deepgaze3
    eval_fixations = fixations[fixations.scanpath_history_length > 3]
    
    eval_fixations = eval_fixations[:10]

    # information_gain = http_model.information_gain(stimuli, eval_fixations, average="image", verbose=True)
    # print("IG:", information_gain)

    for fixation_index in tqdm(range(10)):
        # get server response for one stimulus
        server_density = http_model.conditional_log_density(
            stimulus=stimuli.stimuli[eval_fixations.n[fixation_index]], 
            x_hist=eval_fixations.x_hist[fixation_index], 
            y_hist=eval_fixations.y_hist[fixation_index], 
            t_hist=eval_fixations.t_hist[fixation_index]
        )
        # get test model response
        # test_model_density = test_model(
        #     stimulus=stimuli.stimuli[eval_fixations.n[fixation_index]], 
        #     x_hist=eval_fixations.x_hist[fixation_index], 
        #     y_hist=eval_fixations.y_hist[fixation_index], 
        #     t_hist=eval_fixations.t_hist[fixation_index]   
        # )

        # Testing 
        # test = np.testing.assert_allclose(server_density, test_model_density)
        