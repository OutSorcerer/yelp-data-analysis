FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime

ENV TORCH_HOME=/yelp-data-analysis/.torch_home

WORKDIR /yelp-data-analysis

# TODO: uncomment this before building the final image.
COPY . .

RUN pip install numpy pandas jupyter jupytext pretrainedmodels scikit-learn matplotlib seaborn

CMD jupyter notebook --allow-root --ip 0.0.0.0 --port 8888 --NotebookApp.token=Sy3BMx14nrrYsR9LqzmVcbvHmcGnywyN --no-browser