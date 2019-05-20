# How to run 

Uses image from Docker Hub.
Image contains the entire dataset, prentrained CNN weights, deep photo features and
Jupyter notebook outputs for full reproducibility.

```
docker run --rm -p 8888:8888 sourcerer/yelp-data-analysis
```

And navigate to 

```
http://localhost:8888/notebooks/explore.py?token=Sy3BMx14nrrYsR9LqzmVcbvHmcGnywyN
```
 
in your browser.

To bind to another local port use `-p $CUSTOM_PORT:8888`.

# How to run in debug mode

```
docker run --rm -p 8888:8888 -v $PWD:/yelp-data-analysis yelp-data-analysis
```

Uses local image and mounts current folder as `/yelp-data-analysis` inside container 
to immediately see changes there without rebuilding the container.

# How to rebuild Docker image 

* Clone the repo.
* Manually put `dataset` and `photos` folders downloaded from [the Yelp Open Dataset page](https://www.yelp.com/dataset/) in the root of the repo.
* Run `feature_extractor.py`. That will download Xception model weights and extract deep features for every image.
* Run `build.sh` to build the Docker image. 
