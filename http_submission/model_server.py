from flask import Flask, request, jsonify
import pickle

# Import your model here
from sample_submission import SampleScanpathModel

app = Flask("saliency-model-server")
app.logger.setLevel("DEBUG")

# TODO - replace this with your model
model = SampleScanpathModel()


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_data()
    inputs = pickle.loads(payload)
    app.logger.info(f"Received: {inputs}")

    # TODO - replace this with your model prediction function
    result = model.conditional_log_density(
        inputs["stimulus"],
        inputs["x_hist"],
        inputs["y_hist"],
        inputs["t_hist"],
        inputs["attributes"],
        inputs["out"],
    )
    # resp = pickle.dumps(result)
    app.logger.info(f"Result: {result}")
    # The below assumes that the model returns a numpy array.
    return result.tolist()


def main():
    app.run(host="localhost", port="4000", debug="True", threaded=True)


if __name__ == "__main__":
    main()