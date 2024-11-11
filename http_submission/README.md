# Submission
TODO: Add a description of the submission process here.



## Launching the submission container
TODO: Create a docker-compose file
```bash
cd ./http_submission
docker build -t sample_pysaliency .
```

```bash
docker run --name sample_pysaliency -dp 4000:4000 sample_pysaliency
```
The above command will launch a container named `sample_pysaliency` and expose the port `4000` to the host machine. The container will be running in the background. 

To test the model server, run the sample_evaluation script (Make sure to have the `pysaliency` package installed):
```bash
python ./http_evaluation/sample_evaluation.py
```


To delete the container, run the following command:
```bash
docker stop sample_pysaliency && docker rm sample_pysaliency
```