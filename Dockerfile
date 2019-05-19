FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

WORKDIR /yelp-data-analysis

#COPY . .

RUN pip install numpy pandas jupyter jupytext pretrainedmodels

CMD jupyter notebook --allow-root --ip 0.0.0.0 --port 8888 --NotebookApp.token=Sy3BMx14nrrYsR9LqzmVcbvHmcGnywyN