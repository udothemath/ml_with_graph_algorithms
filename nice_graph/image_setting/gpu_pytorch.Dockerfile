FROM infuseai/docker-stacks:pytorch-notebook-v1-10-2-6dec5b2e-gpu-cuda-11
ENV JUPYTER_ENABLE_LAB=""

#############################
#   APT Package             #
#############################
USER root
COPY requirement-apt.txt $HOME

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3BF863CC \
 && apt-get update \
 && cat $HOME/requirement-apt.txt | xargs apt-get install -y --upgrade --no-install-recommends \
 && apt-get clean \
 && apt-get autoclean

# For PostgreSQL
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        gnupg2 \
        lsb-core \
 && echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
 && wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add - \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        postgresql-12 \
        postgresql-client-12 \
 && apt-get clean \
 && apt-get autoclean

# RUN apt-get update \
#  && apt-get install -y --no-install-recommends \
#         fonts-powerline \
#         powerline

#############################
#   Others                  #
#############################
# USER root

# RUN wget https://github.com/jingweno/ccat/releases/download/v1.1.0/linux-amd64-1.1.0.tar.gz \
#  && tar xfz linux-amd64-1.1.0.tar.gz \
#  && cp linux-amd64-1.1.0/ccat /usr/local/bin/ \
#  && rm linux-amd64-1.1.0.tar.gz

# RUN wget -O /usr/share/tesseract-ocr/4.00/tessdata/chi_tra.traineddata \
#     https://raw.githubusercontent.com/tesseract-ocr/tessdata/master/chi_tra.traineddata

# RUN curl -L http://download.osgeo.org/libspatialindex/spatialindex-src-1.8.5.tar.gz | tar xz \
#  && cd spatialindex-src-1.8.5 \
#  && ./configure \
#  && make \
#  && sudo make install \
#  && sudo ldconfig

# RUN git clone https://github.com/SeanNaren/warp-ctc.git /usr/local/warp-ctc \
#  && git clone https://github.com/cnclabs/smore.git /usr/local/smore && make --directory=/usr/local/smore \
#  && git clone https://github.com/guestwalk/libffm /usr/local/libffm && make --directory=/usr/local/libffm \
#  && git clone https://github.com/pklauke/LibFFMGenerator /usr/local/LibFFMGenerator \
#  && git clone https://github.com/Microsoft/dowhy.git /usr/local/dowhy

#----powerline
# COPY .bash_profile /etc/profile.d/bashrc.sh

###############################
#     VS Code Remote SSH      #
###############################
USER root

ENV NOTVISIBLE "in users profile"
RUN mkdir -p /vat/run/sshd \
 && echo 'jovyan:esun@1313' | chpasswd \
 && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd \
 && echo "export VISIBLE=now" >> /etc/profile \
 && echo "PermitUserEnvironment yes" >> /etc/ssh/sshd_config
 
EXPOSE 22

#############################
#   Conda                   #
#############################
USER $NB_UID
COPY requirement-conda.txt $HOME
RUN conda update --all --yes
RUN conda config --set channel_priority false
RUN conda install --yes --debug --file $HOME/requirement-conda.txt \
 && conda clean -tipsy \
 && fix-permissions $CONDA_DIR

#############################
#         RAPIDS            #
#############################
USER $NB_UID
RUN conda install -c rapidsai rapids
RUN conda clean -tipsy \
 && fix-permissions $CONDA_DIR
#############################
#   Update Python Package   #
#############################
# USER $NB_UID

# RUN conda config --set pip_interop_enabled true

# RUN conda list | \
#     grep "^[^#;]" | \
#     grep -v -P "(^|\s)\Kpython(?=\s|$)" | \
#     grep -v "^jupyter" | \
#     awk '{print $1}' | \
#     xargs conda update --yes \
#  && conda clean -tipsy \
#  && fix-permissions $CONDA_DIR

# RUN pip install pipupgrade \
#  && pipupgrade  --ignore-error --yes

#############################
#   Pypi                    #
#############################
USER $NB_UID

COPY requirement-pip.txt $HOME

RUN pip install --no-cache-dir --user -r $HOME/requirement-pip.txt

# Install fastFM
# RUN git clone --recursive https://github.com/ibayer/fastFM.git \
#  && pip install -r fastFM/requirements.txt \
#  && make -C fastFM/ \
#  && pip install fastFM/

#############################
#   Update                  #
#############################
# RUN conda update --all \
#  && conda install --update-specs --yes\
#        mako==1.1.2

# RUN pip install --upgrade \
#        matplotlib==3.0.2 \
#        urllib3

#############################
#   Jupyter Extension       #
#############################
USER $NB_UID
RUN jupyter nbextension install --py \
        jupyter_dashboards \
        --sys-prefix \
 && jupyter nbextension enable --py jupyter_dashboards --sys-prefix \
 && jupyter nbextension enable --py widgetsnbextension
# RUN jupyter labextension update --all
RUN jupyter labextension install @jupyterlab/hub-extension \
 && jupyter labextension install @jupyterlab/toc \
 && jupyter labextension install @ryantam626/jupyterlab_sublime \
 && jupyter labextension install jupyter-matplotlib \
 && jupyter labextension install jupyter-cytoscape \
 && jupyter labextension install jupyterlab-dash \
 && jupyter labextension install jupyterlab-drawio \
 && jupyter labextension install nbdime-jupyterlab \
 && jupyter lab clean -y 
RUN npm cache clean --force \ 
 && rm -rf /home/$NB_USER/.cache/yarn \
 && rm -rf /home/$NB_USER/.node-gyp 
RUN fix-permissions $CONDA_DIR \
 && fix-permissions /home/$NB_USER

RUN pip install --no-cache-dir nbresuse \
 && jupyter serverextension enable --py nbresuse \
 && jupyter lab clean -y

#############################
#   Julia                   #
#############################
# USER root

# RUN wget --directory-prefix=/usr/local/lib/ https://julialang-s3.julialang.org/bin/linux/x64/1.1/julia-1.1.0-linux-x86_64.tar.gz \
#  && tar -xvf /usr/local/lib/julia-1.1.0-linux-x86_64.tar.gz -C /usr/local/lib/ \
#  && rm /usr/local/lib/julia-1.1.0-linux-x86_64.tar.gz \
#  && chown -R jovyan:users /usr/local/lib/julia-1.1.0/

# ENV PATH "/usr/local/lib/julia-1.1.0/bin:$PATH"
# ENV JULIA_DEPOT_PATH "/usr/local/lib/julia-1.1.0/"
# ENV JUPYTER "/opt/conda/bin/jupyter-labhub"

# RUN julia -e 'using Pkg; Pkg.add("PyPlot"); Pkg.build("PyPlot"); \
#         Pkg.add("IJulia"); Pkg.build("IJulia"); \
#         Pkg.add("NetCDF"); Pkg.build("NetCDF"); \
#         Pkg.add("MAT"); Pkg.build("MAT")' \
#  && chown -R jovyan:users /usr/local/lib/julia-1.1.0/

#############################
#   Tensorflow Debug        #
#############################
# USER $NB_UID

# RUN pip install --no-cache-dir \
#         keras-applications==1.0.6 \
#         keras-preprocessing==1.0.6 \
#         tensorflow-gpu==1.13.1

#############################
#   Tesseract Package       #
#############################
# USER root
# RUN apt-get update \
#  && apt-get install -y --no-install-recommends software-properties-common \
#  && echo "deb http://ppa.launchpad.net/alex-p/tesseract-ocr/ubuntu bionic main" >> /etc/apt/source.list \
#  && echo "deb-src http://ppa.launchpad.net/alex-p/tesseract-ocr/ubuntu bionic main" >> /etc/apt/source.list \
#  && add-apt-repository --yes ppa:alex-p/tesseract-ocr \
#  && apt-get update \
#  && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-chi-tra

#########################
#   Docker CLI          #
#########################
USER root
RUN apt-get install -y software-properties-common
RUN apt-get update
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - \
 && add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable" \
 && apt-get update \
 && apt-get install -y --no-install-recommends docker-ce-cli

#############################
#       RPA Package         #
#############################
# USER root
# RUN wget https://chromedriver.storage.googleapis.com/87.0.4280.88/chromedriver_linux64.zip \
#  && unzip chromedriver_linux64.zip \
#  && chmod +x chromedriver \
#  && mv chromedriver /usr/bin/ \
#  && rm chromedriver_linux64.zip

#############################
#       Python3.8 & 3.6     #
#############################
USER root 
RUN add-apt-repository -y ppa:deadsnakes/ppa \
 && apt update 
RUN apt install -y python3.8 \
 && apt-get install -y python3.8-venv 
RUN apt install -y python3.6 \
 && apt-get install -y python3.6-venv
#############################
#    Redis Stack Server     #
#############################
USER root
RUN curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list \
 && apt-get update \
 && apt-get install -y redis-stack-server \
 && pip install redis-server
#############################
#          Neo4j            #
#############################
USER root
RUN wget -O - https://debian.neo4j.com/neotechnology.gpg.key | apt-key add - \
 && echo 'deb https://debian.neo4j.com stable latest' | tee /etc/apt/sources.list.d/neo4j.list \
 && apt update
RUN apt install -y neo4j=1:4.4.6
RUN wget https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/4.4.0.6/apoc-4.4.0.6-all.jar \
 && cp apoc-4.4.0.6-all.jar /var/lib/neo4j/plugins/ \
 && chown neo4j:neo4j /var/lib/neo4j/plugins/apoc-4.4.0.6-all.jar
#############################
#   Pytorch GNN Libraries   #
#############################
USER $NB_UID
# RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.2+cu111.html
RUN pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html

#############################
# For PrimeHub Schedule Job #
#############################
# USER $NB_UID

# ENV PATH $PATH:/opt/conda/bin
# ENV PIP_CONFIG_FILE /etc/python/pip.config

# COPY ai-cloud-pip.config /etc/python/pip.config


# TODO:
# 0. change base imagee to pytorch notebook
# pytorch notebook has cuda 11.2, pytorch 1.10.2+cu111
# 1. check pytorch version and check its cuda version, 
# make sure it is the same as nvcc --version 
# -> OK: cuda 11.2 is downward compatible to pytorch 1.10.2 + cu111
# 2. make sure nvcc --version support GPU driver version of aicloud 
# -> OK: cuda 11.2 is supported by AiCloud (not MLaaS!?)
# 3. Make sure PYG support the current pytorch & cuda 
# `pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.2+cu111.html`
# 4. Make sure Dgl support the current pytorch & cuda 
# `pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html`
# -> Pytorch OK: pytorch 1.10.2 > pytorch 1.9.0+ 
# -> CUDA OK: CUDA 11.2 support dgl-cu111
# (X: Not working) 5. Make sure rapids support the current cuda & python
# `conda create -n rapids -c rapidsai -c nvidia -c conda-forge rapids=22.06 python=3.8 cudatoolkit=11.0`

