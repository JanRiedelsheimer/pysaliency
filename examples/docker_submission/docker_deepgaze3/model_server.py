from flask import Flask, request
# from flask_orjson import OrjsonProvider
import numpy as np
import json
from PIL import Image
from io import BytesIO
import orjson
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch

# Import your model here
import deepgaze_pytorch

# Flask server
app = Flask("saliency-model-server")
# app.json_provider = OrjsonProvider(app)
app.logger.setLevel("DEBUG")

# # TODO - replace this with your model
model = deepgaze_pytorch.DeepGazeIII(pretrained=True)

@app.route('/conditional_log_density', methods=['POST'])
def conditional_log_density():
    # get data
    data = json.loads(request.form['json_data'])

    # extract scanpath history
    x_hist = np.array(data['x_hist'])
    y_hist = np.array(data['y_hist'])
    # t_hist = np.array(data['t_hist'])
    # attributes = data.get('attributes', {})

    # extract stimulus
    image_bytes = request.files['stimulus'].read()
    image = Image.open(BytesIO(image_bytes))
    stimulus = np.array(image)

    # centerbias for deepgaze3 model
    centerbias_template = np.zeros((1024, 1024)) # (1024, 1024)
    centerbias = zoom(centerbias_template, 
                        (stimulus.shape[0]/centerbias_template.shape[0], 
                         stimulus.shape[1]/centerbias_template.shape[1]), 
                        order=0,
                        mode='nearest'
    )  
    centerbias -= logsumexp(centerbias)

    # make tensors for deepgaze3 model
    image_tensor = torch.tensor([stimulus.transpose(2, 0, 1)])
    centerbias_tensor = torch.tensor([centerbias])
    x_hist_tensor = torch.tensor([x_hist[model.included_fixations]]) 
    y_hist_tensor = torch.tensor([y_hist[model.included_fixations]])

    # return model response
    log_density = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
    log_density_list = log_density.tolist()
    response = orjson.dumps({'log_density': log_density_list})
    return response


@app.route('/type', methods=['GET'])
def type():
    type = "ScanpathModel"
    version = "v1.0.0"
    return orjson.dumps({'type': type, 'version': version})


   

def main():
    app.run(host="localhost", port="4000", debug="True", threaded=True)


if __name__ == "__main__":
    main()