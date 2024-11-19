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
        self.log_density_url = url + "/conditional_log_density"
        self.type_url = url + "/type"




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
        response = requests.post(f"{self.log_density_url}", data={'json_data': json.dumps(json_data)}, files={'stimulus': image_bytes.getvalue()})

        # parse response
        if response.status_code != 200:
            raise ValueError(f"Server returned status code {response.status_code}")

        return np.array(response.json()['log_density'])

    def type(self):
        response = requests.get(f"{self.type_url}")
        return np.array(response.json())


if __name__ == "__main__":
    http_model = HTTPScanpathModel("http://localhost:4000")
    type = http_model.type()

    # for testing
    model = MySimpleScanpathModel()

    # get MIT1003 dataset
    stimuli, fixations = pysaliency.get_mit1003(location='pysaliency_datasets')
    # fixation_index = 32185
    fixation_index = 2
    # density_list = []
    # version_list = []
    # for fixation_index in range(1000):

    # get server response for one stimulus
    server_density = http_model.conditional_log_density(
        stimulus=stimuli.stimuli[fixations.n[fixation_index]], 
        x_hist=fixations.x_hist[fixation_index], 
        y_hist=fixations.y_hist[fixation_index], 
        t_hist=fixations.t_hist[fixation_index]
    )
    model_density = model.conditional_log_density(
        stimulus=stimuli.stimuli[fixations.n[fixation_index]], 
        x_hist=fixations.x_hist[fixation_index], 
        y_hist=fixations.y_hist[fixation_index], 
        t_hist=fixations.t_hist[fixation_index]
        
    )
    # get server type
    print(type)

    # Testing 

    test = np.testing.assert_allclose(server_density, model_density)
    print(test)