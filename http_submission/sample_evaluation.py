import numpy as np
import pickle
import pysaliency
import requests


class HTTPScanpathModel(pysaliency.ScanpathModel):
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