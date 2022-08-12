FROM jupyter/base-notebook:python-3.8.3
ENV JUPYTER_ENABLE_LAB=""

#############################
#   APT Package             #
#############################
USER root
COPY requirement-apt.txt $HOME
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

# For CUDA 11.2
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys A4B469963BF863CC
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub \
 && sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list' \
 && apt-get update \
 && apt-get --yes install cuda-toolkit-11-2 \
 && sh -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-machine-learning.list' \
 && apt-get update \
 && apt-get install --yes --no-install-recommends cuda-11-2 libcudnn8=8.1.0.77-1+cuda11.2 libcudnn8-dev=8.1.0.77-1+cuda11.2 \ 
 && apt-get clean \
 && apt-get autoclean

# RUN apt-get update \
#  && apt-get install -y --no-install-recommends \
#       fonts-powerline \
#       powerline
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
USER root
RUN conda install -c conda-forge mamba
COPY requirement-conda.txt $HOME
RUN mamba install -c anaconda -c conda-forge --yes --file $HOME/requirement-conda.txt
#############################
#   RAPIDS TOOLS            #
#############################
RUN mamba install -c rapidsai rapidsai:rapids --yes
RUN mamba install -c rapidsai -c conda-forge dask-sql --yes
RUN mamba install -c rapidsai -c conda-forge graphistry --yes
RUN mamba install -c rapidsai -c conda-forge dash --yes
RUN mamba clean --all --yes
RUN fix-permissions $CONDA_DIR

#############################
#   Pypi                    #
#############################
USER $NB_UID

COPY requirement-pip.txt $HOME

RUN pip --default-timeout=10000 install --no-cache-dir -r $HOME/requirement-pip.txt

#############################
#   Jupyter Extension       #
#############################
USER $NB_UID
RUN jupyter nbextension install --py \
        jupyter_dashboards \
        --sys-prefix \
 && jupyter nbextension enable --py jupyter_dashboards --sys-prefix \
 && jupyter nbextension enable --py widgetsnbextension 
RUN jupyter labextension install @jupyterlab/hub-extension \
 && jupyter labextension install @jupyterlab/toc \
 && jupyter labextension install @ryantam626/jupyterlab_sublime \
 && jupyter labextension install jupyter-matplotlib \
 && jupyter labextension install jupyter-cytoscape \
 && jupyter labextension install jupyterlab-dash \
 && jupyter labextension install jupyterlab-drawio \
 && jupyter labextension install nbdime-jupyterlab 
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
#          Neo4j            #
#############################
USER root
RUN wget -O - https://debian.neo4j.com/neotechnology.gpg.key | apt-key add - \
 && echo 'deb https://debian.neo4j.com stable latest' | tee /etc/apt/sources.list.d/neo4j.list \
 && apt update \
 && apt install -y neo4j=1:4.4.6 \
 && wget https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/4.4.0.6/apoc-4.4.0.6-all.jar \
 && cp apoc-4.4.0.6-all.jar /var/lib/neo4j/plugins/ \
 && chown neo4j:neo4j /var/lib/neo4j/plugins/apoc-4.4.0.6-all.jar
RUN fix-permissions /var/lib/neo4j \
 && fix-permissions /var/log/neo4j
#############################
#   Pytorch GNN Libraries   #
#############################
USER $NB_UID
RUN pip install torch==1.10.2 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111
RUN pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.2+cu111.html
RUN pip install --no-cache-dir dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html
#############################
#   RAPIDS                  #
#############################
RUN mamba install -c rapidsai -c nvidia -c conda-forge cudf=22.06 cuml=22.06 cugraph=22.06 \
 && cuspatial=22.06 cuxfilter=22.06 cusignal=22.06 cucim=22.06
RUN mamba install -c rapidsai -c nvidia -c conda-forge cudatoolkit=11.2
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib
RUN export PATH=$PATH:/usr/local/cuda/bin
#############################
#   Other ETL tools         #
#############################
RUN pip install --no-cache-dir vaex modin[all]
RUN pip install 'polars[pyarrow]'
RUN pip install swifter
#############################
#      Other APT tools      #
#############################
USER root
RUN apt update
RUN apt install -y git vim lsof dkms aptitude
RUN dkms status
#############################
# Downgrade Nvidia-driver   #
#############################
RUN apt -y autoremove --purge nvidia-driver-515 \
 && apt -y clean \
 && add-apt-repository ppa:graphics-drivers/ppa \
 && apt update 
RUN aptitude install -y nvidia-driver-470 nvidia-dkms-470 \
 && aptitude install -y nvidia-driver-470=470.42.01-0ubuntu1 \
 && aptitude install -y nvidia-dkms-470=470.42.01-0ubuntu1
RUN dkms status
###############################
# For PrimeHub Job Submission #
###############################
USER $NB_UID

ENV PATH $PATH:/opt/conda/bin
ENV PIP_CONFIG_FILE /etc/python/pip.config

COPY ai-cloud-pip.config /etc/python/pip.config
