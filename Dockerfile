FROM tensorflow/tensorflow:1.15.4-gpu-py3
LABEL name=popcorn \
      version=0.1 \
      maintainer=reda-abdellah.kamraoui@labri.fr \
      net.volbrain.pipeline.mode=gpu_only \
      net.volbrain.pipeline.name=popcorn
RUN curl -LO http://ssd.mathworks.com/supportfiles/downloads/R2017b/deployment_files/R2017b/installers/glnxa64/MCR_R2017b_glnxa64_installer.zip && \
    mkdir MCR && \
    cp MCR_R2017b_glnxa64_installer.zip MCR && \
    cd MCR && \
    unzip MCR_R2017b_glnxa64_installer.zip && \
    ./install -mode silent -agreeToLicense yes && \
    cd .. && \
    rm -rf MCR
ARG DEBIAN_FRONTEND=noninteractive
ARG DEBCONF_NONINTERACTIVE_SEEN=true
WORKDIR /opt/popcorn
ENV LD_LIBRARY_PATH=/usr/local/MATLAB/MATLAB_Runtime/v93/runtime/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v93/bin/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v93/sys/os/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v93/sys/opengl/lib/glnxa64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV XAPPLRESDIR=/usr/local/MATLAB/MATLAB_Runtime/v93/X11/app-defaults
ENV MCR_CACHE_VERBOSE=true
ENV MCR_CACHE_ROOT=/tmp
RUN mkdir /usr/local/MATLAB/MATLAB_Runtime/v93/sys/os/glnxa64/exclude
RUN mkdir /usr/local/MATLAB/MATLAB_Runtime/v93/bin/glnxa64/exclude
RUN mv /usr/local/MATLAB/MATLAB_Runtime/v93/sys/os/glnxa64/libstdc++.so.6.* /usr/local/MATLAB/MATLAB_Runtime/v93/sys/os/glnxa64/exclude/
RUN mv /usr/local/MATLAB/MATLAB_Runtime/v93/bin/glnxa64/libfreetype* /usr/local/MATLAB/MATLAB_Runtime/v93/bin/glnxa64/exclude/
RUN echo deb http://archive.ubuntu.com/ubuntu/ trusty main restricted universe multiverse  >> /etc/apt/sources.list
RUN echo deb http://archive.ubuntu.com/ubuntu/ trusty-security main restricted universe multiverse  >> /etc/apt/sources.list
RUN echo deb http://archive.ubuntu.com/ubuntu/ trusty-updates main restricted universe multiverse  >> /etc/apt/sources.list
RUN echo deb http://archive.ubuntu.com/ubuntu/ trusty-proposed main restricted universe multiverse  >> /etc/apt/sources.list
RUN echo deb http://archive.ubuntu.com/ubuntu/ trusty-backports main restricted universe multiverse  >> /etc/apt/sources.list
RUN apt-get update
RUN apt -qqy install libx11-dev xserver-xorg libfontconfig1 libxt6 libxcomposite1 libasound2 libxext6 texlive-xetex
RUN apt -qqy install wget

RUN wget -q https://github.com/Inria-Visages/Anima-Public/releases/download/v4.0.1/Anima-Ubuntu-4.0.1.zip
RUN unzip Anima-Ubuntu-4.0.1.zip
RUN git lfs install
RUN git clone --depth 1 https://github.com/Inria-Visages/Anima-Scripts-Data-Public.git
RUN mkdir /root/.anima/
RUN mkdir /tmp/test_image

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1FkcruF4AeJjgmcVpfKQgbrMXOvG2bO7D' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1FkcruF4AeJjgmcVpfKQgbrMXOvG2bO7D" -O Compilation_lesionBrain_v11_fullpreprocessing.zip && rm -rf /tmp/cookies.txt
RUN unzip Compilation_lesionBrain_v11_fullpreprocessing.zip
RUN mv Compilation_lesionBrain_v11_fullpreprocessing/* /opt/popcorn

RUN pip3 install scikit-learn statsmodels  keras==2.2.4 pillow nibabel==2.5.2 scikit-image==0.17.2
RUN mkdir /Weights

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-IVxeFcdM_h3zH5q_gGeXr17UqhvHdpa' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-IVxeFcdM_h3zH5q_gGeXr17UqhvHdpa" -O vol_trained_all.zip && rm -rf /tmp/cookies.txt
RUN unzip vol_trained_all.zip
RUN mv *.pt /Weights/

RUN apt -qqy install git
RUN chmod 777 -R /opt/popcorn/*

RUN mv DLB_docker/* /opt/popcorn
RUN mkdir /data/
RUN mkdir -p /data/patients/patient_X/

RUN git clone https://github.com/Reda-Abdellah/popcorn_docker.git
RUN cp -avr popcorn_docker/Anima-Scripts-Public /anima/Anima-Scripts-Public
RUN cp popcorn_docker/config.txt /root/.anima
RUN cp popcorn_docker/*.py /opt/popcorn/
RUN cp -avr popcorn_docker/Registration /opt/popcorn/Registration
