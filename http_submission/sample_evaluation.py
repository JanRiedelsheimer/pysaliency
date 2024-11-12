import numpy as np
import pickle
import requests
import sys
from sample_submission import MySimpleScanpathModel
from PIL import Image
from io import BytesIO
import json
import matplotlib.pyplot as plt
from pysaliency.plotting import plot_scanpath
sys.path.insert(0, '..')
import pysaliency

class HTTPScanpathModel(MySimpleScanpathModel):
    def __init__(self, url):
        self.url = url

    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):

        # build request
        pil_image = Image.fromarray(stimulus)
        image_bytes = BytesIO()
        pil_image.save(image_bytes, format='png')

        def _convert_attribute(attribute):
            if isinstance(attribute, np.ndarray):
                return attribute.tolist()
            return attribute

        json_data = {
            "x_hist": list(x_hist),
            "y_hist": list(y_hist),
            "t_hist": list(t_hist),
            "attributes": {key: _convert_attribute(value) for key, value in (attributes or {}).items()}
        }

        # send request
        response = requests.post(f"{self.url}", data={'json_data': json.dumps(json_data)}, files={'stimulus': image_bytes.getvalue()})

        # parse response
        if response.status_code != 200:
            raise ValueError(f"Server returned status code {response.status_code}")

        return np.array(response.json()['log_density'])

if __name__ == "__main__":
    http_model = HTTPScanpathModel("http://localhost:4000/conditional_log_density")

    # get MIT1003 dataset
    stimuli, fixations = pysaliency.get_mit1003(location='pysaliency_datasets')
    fixation_index = 32185
    # get server response for one stimulus
    server_response = http_model.conditional_log_density(
        stimulus=stimuli.stimuli[fixations.n[fixation_index]], 
        x_hist=fixations.x_hist[fixation_index], 
        y_hist=fixations.y_hist[fixation_index], 
        t_hist=fixations.t_hist[fixation_index]
    )
    # TODO: delete plotting part
    # plot server response, only for testing

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].set_axis_off()
    axs[1].set_axis_off()

    axs[0].imshow(stimuli.stimuli[fixations.n[fixation_index]])
    plot_scanpath(stimuli, fixations, fixation_index, visualize_next_saccade=True, ax=axs[0])
    axs[0].set_title("Image")

    axs[1].imshow(server_response)

    axs[1].set_title("http_model_log_density")
    fig.savefig("test.png")