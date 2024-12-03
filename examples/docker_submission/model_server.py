from flask import Flask, request, jsonify
import numpy as np
import json
from PIL import Image
from io import BytesIO
# import pickle

# Import your model here
from sample_submission import MySimpleScanpathModel

app = Flask("saliency-model-server")
app.logger.setLevel("DEBUG")

# # TODO - replace this with your model
model = MySimpleScanpathModel()


@app.route('/conditional_log_density', methods=['POST'])
def conditional_log_density():
    data = json.loads(request.form['json_data'])
    x_hist = np.array(data['x_hist'])
    y_hist = np.array(data['y_hist'])
    t_hist = np.array(data['t_hist'])
    attributes = data.get('attributes', {})

    image_bytes = request.files['stimulus'].read()
    image = Image.open(BytesIO(image_bytes))
    stimulus = np.array(image)

    log_density = model.conditional_log_density(stimulus, x_hist, y_hist, t_hist, attributes)
    return jsonify({'log_density': log_density.tolist()})


@app.route('/type', methods=['GET'])
def type():
    type = "ScanpathModel"
    version = "v1.0.0"
    return jsonify({'type': type, 'version': version})
   

def main():
    app.run(host="localhost", port="4000", debug="True", threaded=True)


if __name__ == "__main__":
    main()