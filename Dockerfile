FROM nvidia/cuda:11.0.3-base-ubuntu20.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    wget \
                    bzip2 \
                    ca-certificates \
                    build-essential \
                    autoconf \
                    libtool \
                    pkg-config \
                    git && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p /usr/local/miniconda3 && \
    rm -rf ~/miniconda3/miniconda.sh 

ENV PATH="/usr/local/miniconda3/bin/":${PATH}

RUN conda config --add channels conda-forge

## install pytorch
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

## install monai and dependencies
RUN pip install monai==1.3.0
RUN pip install "monai[einops]"
RUN pip install nilearn scikit-image

## pull trained weights and atlases
RUN cd / && wget http://users.bmap.ucla.edu/~yeunkim/rodentpipeline/pretrained_weights_rodseg.tar.gz && \
    tar xzvf pretrained_weights_rodseg.tar.gz && rm /pretrained_weights_rodseg.tar.gz

ENV PTWTS="/mod5/"

COPY . /mousebrainextractor

RUN chmod +x /mousebrainextractor/bin/*

ENTRYPOINT ["/mousebrainextractor/bin/run_mbe_predict_skullstrip_container.py"]