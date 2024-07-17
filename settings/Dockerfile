FROM ubuntu:18.04

ENV WORKDIR=/e2sc-is
ENV PYTHONIOENCODING utf-8
ENV LANG="C.UTF-8"
ENV JOBLIB_TEMP_FOLDER=/tmp
ENV TMPDIR=/tmp
ENV DEBIAN_FRONTEND=teletype
ENV TZ=America/Sao_Paulo

WORKDIR $WORKDIR
COPY . /e2sc-is/settings/

#echo "Installing python..."
RUN apt update -y --fix-missing \
                    && apt install -y --no-install-recommends apt-utils software-properties-common build-essential \
                    && add-apt-repository -y ppa:deadsnakes/ppa \
                    && apt update \
                    && apt install -y python3.6 python3.6-dev python3-pip \
                    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
                    apt-get install -y --no-install-recommends wget nano curl git ninja-build ccache libopenblas-dev libopencv-dev

#echo "Linking python..."
RUN ln -sfn /usr/bin/python3.6 /usr/bin/python3 && \
    ln -sfn /usr/bin/python3 /usr/bin/python && \
    ln -sfn /usr/bin/pip3 /usr/bin/pip

#echo "Upgrading setuptools pip wheel" and "Installing requeriments"
RUN python -m pip install --upgrade setuptools pip wheel \
    && python -m pip install -r /e2sc-is/settings/requirements.txt 
#Install
#cd e2sc-is/settings/
#docker build -t e2sc:1.0 .
#Using
#cd ..
#docker run --rm --name e2sctest -v .:/e2sc-is -i -t e2sc:0.1 /bin/bash