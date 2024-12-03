from .models import ScanpathModel
from PIL import Image
from io import BytesIO
import requests
import json
import numpy as np

class HTTPScanpathModel(ScanpathModel):
    def __init__(self, url):
        self.url = url
        self.check_type()

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
        # TODO in constuctor aufrufen, damit direkt gecheckt wird ob type passt
        response = requests.get(f"{self.type_url}").json()
        if not response['type'] == 'ScanpathModel':
            raise ValueError(f"invalid Model type: {response['type']}. Expected 'ScanpathModel'")
        if not response['version'] in ['v1.0.0']:
            raise ValueError(f"invalid Model type: {response['version']}. Expected 'v1.0.0'")
