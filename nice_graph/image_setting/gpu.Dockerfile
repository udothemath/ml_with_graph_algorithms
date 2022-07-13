FROM infuseai/docker-stacks:tensorflow-notebook-v2-5-0-63fdf50a-gpu-cuda-11
# FROM infuseai/docker-stacks:pytorch-notebook-v1-10-2-6dec5b2e-gpu-cuda-11
ENV JUPYTER_ENABLE_LAB=""

#############################
#   APT Package             #
#############################
USER root
COPY requirement-apt.txt $HOME
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update \
 && cat $HOME/requirement-apt.txt | xargs apt-get install -y --upgrade --no-install-recommends \
 && apt-get clean \
 && apt-get autoclean


# For PostgreSQL
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        gnupg2 \
        lsb-core \
        gpg-agent \
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
#       fonts-powerline \
#       powerline

#############################
#   Others                  #
#############################
USER root

RUN wget -O /usr/share/tesseract-ocr/4.00/tessdata/chi_tra.traineddata \
    https://raw.githubusercontent.com/tesseract-ocr/tessdata/master/chi_tra.traineddata \
 && curl -L http://download.osgeo.org/libspatialindex/spatialindex-src-1.8.5.tar.gz | tar xz \
 && cd spatialindex-src-1.8.5 \
 && ./configure \
 && make \
 && sudo make install \
 && sudo ldconfig \
 && git clone https://github.com/SeanNaren/warp-ctc.git /usr/local/warp-ctc \
 && git clone https://github.com/cnclabs/smore.git /usr/local/smore && make --directory=/usr/local/smore \
 && git clone https://github.com/guestwalk/libffm /usr/local/libffm && make --directory=/usr/local/libffm \
 && git clone https://github.com/pklauke/LibFFMGenerator /usr/local/LibFFMGenerator \
 && git clone https://github.com/Microsoft/dowhy.git /usr/local/dowhy

#----powerline
COPY .bash_profile /etc/profile.d/bashrc.sh

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
RUN conda update --all
RUN conda install --quiet --yes --file $HOME/requirement-conda.txt \
 && conda clean -tipsy \
 && fix-permissions $CONDA_DIR

#############################
#   Update Python Package   #
#############################
# USER $NB_UID

# RUN conda config --set pip_interop_enabled true

# RUN conda list --name base | \
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

RUN pip install --no-cache-dir -r $HOME/requirement-pip.txt

# Install fastFM
RUN git clone --recursive https://github.com/ibayer/fastFM.git \
 && pip install -r fastFM/requirements.txt \
 && make -C fastFM/ \
 && pip install fastFM/

#############################
#   Jupyter Extension       #
#############################
USER $NB_UID
RUN jupyter nbextension install --py \
        jupyter_dashboards \
        --sys-prefix \
 && jupyter nbextension enable --py jupyter_dashboards --sys-prefix \
 && jupyter nbextension enable --py widgetsnbextension
RUN jupyter labextension install @jupyterlab/hub-extension
RUN jupyter labextension install @jupyterlab/toc
RUN jupyter labextension install @ryantam626/jupyterlab_sublime
RUN jupyter labextension install jupyter-matplotlib
RUN jupyter labextension install jupyter-cytoscape
RUN jupyter labextension install jupyterlab-dash
RUN jupyter labextension install jupyterlab-drawio
RUN jupyter labextension install nbdime-jupyterlab
RUN jupyter lab clean -y \
 && npm cache clean --force \ 
 && rm -rf /home/$NB_USER/.cache/yarn \
 && rm -rf /home/$NB_USER/.node-gyp 
RUN fix-permissions $CONDA_DIR \
 && fix-permissions /home/$NB_USER

RUN pip install --no-cache-dir nbresuse \
 && jupyter serverextension enable --py nbresuse \
 && jupyter lab clean -y

#########################
#   Docker CLI          #
#########################
USER root
RUN apt install -y software-properties-common \
 && add-apt-repository ppa:libreoffice/ppa \
 && apt update
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - \
 && add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable" \
 && apt-get update \
 && apt-get install -y --no-install-recommends docker-ce-cli

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
USER $NB_UID
RUN fix-permissions /var/lib/neo4j \
 && fix-permissions /var/log/neo4j
#############################
#   Pytorch GNN Libraries   #
#############################
USER $NB_UID
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
RUN pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
##############################
# For PrimeHub Job Submission#
##############################
USER $NB_UID

ENV PATH $PATH:/opt/conda/bin
ENV PIP_CONFIG_FILE /etc/python/pip.config

COPY ai-cloud-pip.config /etc/python/pip.config
