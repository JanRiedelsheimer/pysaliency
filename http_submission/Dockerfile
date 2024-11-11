# Specify a base image depending on the project.
FROM bitnami/python:3.8
# For more complex examples, might need to use a different base image.
# FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

WORKDIR /app

ENV HTTP_PORT=4000

RUN apt-get update \
    && apt-get -y install gcc

COPY ./requirements.txt ./
RUN python -m pip install -U pip \
    && python -m pip install -r requirements.txt 

COPY . ./

# This is needed for Singularity builds.
EXPOSE $HTTP_PORT

# The entrypoint for a container,
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:4000", "--pythonpath", ".", "model_server:app"]