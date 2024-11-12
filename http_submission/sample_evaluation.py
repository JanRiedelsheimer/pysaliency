import numpy as np
import pickle
import requests
import sys
from sample_submission import SampleScanpathModel
sys.path.insert(0, '..')
import pysaliency

class HTTPScanpathModel(SampleScanpathModel):
    def __init__(self, url):
        self.url = url

    def conditional_log_density(
        self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None
    ):
        inputs = {
            "stimulus": stimulus,
            "x_hist": x_hist,
            "y_hist": y_hist,
            "t_hist": t_hist,
            "attributes": attributes,
            "out": out,
        }
        payload = pickle.dumps(inputs)
        response = requests.post(self.url, data=payload)
        # print(f"Received: {response.json()}")
        return np.array(response.json())

# class HTTPScanpathModel(pysaliency.ScanpathModel):
#     def __init__(self, url):
#         self.url = url

#     def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
#         # build request
#         pil_image = Image.fromarray(stimulus)
#         image_bytes = BytesIO()
#         pil_image.save(image_bytes, format='png')

#         def _convert_attribute(attribute):
#             if isinstance(attribute, np.ndarray):
#                 return attribute.tolist()
#             return attribute

#         json_data = {
#             "x_hist": list(x_hist),
#             "y_hist": list(y_hist),
#             "t_hist": list(t_hist),
#             "attributes": {key: _convert_attribute(value) for key, value in (attributes or {}).items()}
#         }

#         # send request

#         response = requests.post(f"{self.url}/conditional_log_density", json=json_data, files={'stimulus': image_bytes.getvalue()})

#         # parse response

#         if response.status_code != 200:
#             raise ValueError(f"Server returned status code {response.status_code}")

#         return np.array(response.json()['log_density'])

if __name__ == "__main__":
    http_model = HTTPScanpathModel("http://localhost:4000/predict")

    print(
        http_model.conditional_log_density(
            [1, 1.4, 10, 1],
            [1, 1, 0.51, 1],
            [1, 1, 2, 1],
            [1, 3, 1, 1],
        )
    )