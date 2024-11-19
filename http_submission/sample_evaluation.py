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

    @property
    def log_density_url(self):
        return self.url + "/conditional_log_density"
    
    @property
    def type_url(self):
        return self.url + "/type"
    
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

    def check_type(self):
        response = requests.get(f"{self.type_url}").json()
        if not response['type'] == 'ScanpathModel':
            raise ValueError(f"invalid Model type: {response['type']}. Expected 'ScanpathModel'")
        if not response['version'] in ['v1.0.0']:
            raise ValueError(f"invalid Model type: {response['version']}. Expected 'v1.0.0'")


if __name__ == "__main__":
    http_model = HTTPScanpathModel("http://localhost:4000")
    http_model.check_type()

    # for testing
    model = MySimpleScanpathModel()

    # get MIT1003 dataset
    stimuli, fixations = pysaliency.get_mit1003(location='pysaliency_datasets')

    eval_fixations = fixations[fixations.scanpath_history_length > 0]
    server_density_list = []
    model_density_list = []
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

        server_density_list.append(server_density)
        model_density_list.append(model_density)

    # Testing 
    test = np.testing.assert_allclose(server_density_list, model_density_list)
    print(test)