# Loading DeepGA

Create container to develop

```
docker build --tag deepga/dev_img_covid:24.dev -f Dockerfile.dev .

```

Run image in a container
```
docker run --gpus all -it -d --name dev_img_covid --network=net-deepga -v ${PWD}:/develop deepga/dev_img_covid:24.dev
```
