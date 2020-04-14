FROM nvidia/cuda:10.0-base-ubuntu16.04
FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN pip install opencv-python
RUN pip install gym[atari]
RUN pip install tensorboardX
RUN apt update
RUN apt install -y libsm6 libxext6 libxrender-dev

WORKDIR /app
