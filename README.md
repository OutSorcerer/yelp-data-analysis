# How to run 

Uses image from Docker Hub.

```
docker run --rm -p 8888:8888 sourcerer/yelp-data-analysis
```

And navigate to `http://localhost:8888?token=Sy3BMx14nrrYsR9LqzmVcbvHmcGnywyN`.

To bind to another local port use `-p $CUSTOM_PORT:8888`.

# How to run in debug mode

```
docker run --rm -p 8888:8888 -v $PWD:/yelp-data-analysis yelp-data-analysis
```

Uses local image and mount current folder as `/yelp-data-analysis` inside container 
to immediately see changes there without rebuilding the container.

# How to rebuild Docker image 

* Clone the repo
* Manually put `dataset` and `photos` in the root of the repo
* Run `build.sh` 
