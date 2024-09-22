# Loading DeepGA

Create container to develop

```
docker build --tag deep/dev_img_covid:24.dev -f Dockerfile.dev .

```

Run image in a container
```
docker run --gpus all -it -d --name dev_img_covid --network=net-deep -v ${PWD}:/develop deep/dev_img_covid:24.dev
```


## To run jupyter container 

```
docker build --tag deep/jupyter.cuda:24.09 -f Dockerfile.jupyter .
```

Run image in a container
```
docker run --gpus all -it -d --name jupyter -p 8888:8888 --network=net-deep -v ${PWD}/work:/home/ubuntu/work deep/jupyter.cuda:24.09
```
