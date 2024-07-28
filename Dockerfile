# docker run -it --rm -p 8888:8888 --gpus all 
FROM quay.io/jupyter/tensorflow-notebook:cuda-latest
WORKDIR /code

RUN pip install torch torchvision torchaudio torchtext
RUN pip install pandas xlwt torchsummary torchvision
RUN pip install pytorch-forecasting



# docker build --tag phen/notebook.cuda:24.05 .

# docker run -it --rm -p 8888:8888 --gpus all --name notebook.cuda 