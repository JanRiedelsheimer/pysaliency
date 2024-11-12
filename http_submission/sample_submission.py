import numpy as np
# import pysaliency

class SampleScanpathModel():
    def __init__(self):
        super().__init__()

    def conditional_log_density(self, stimulus, x_hist, y_hist, t_hist, attributes=None, out=None):
        return np.log(stimulus)

# from io import BytesIO

# import pysaliency
# import requests
# from PIL import Image
# import numpy as np


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