FROM rapidsai/rapidsai:22.06-cuda11.2-runtime-ubuntu18.04-py3.8
ENV JUPYTER_ENABLE_LAB=""

#############################
#   APT Package             #
#############################
USER root
COPY requirement-apt.txt $HOME
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3BF863CC
RUN apt-get update \
 && cat $HOME/requirement-apt.txt | xargs apt-get install -y --upgrade --no-install-recommends \
 && apt-get clean \
 && apt-get autoclean



# For PostgreSQL
USER root
RUN apt-get update 
RUN apt-get install -y --no-install-recommends gnupg2 
RUN apt-get install -y --no-install-recommends lsb-core
RUN apt-get install -y --no-install-recommends gpg-agent
RUN echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
 && wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - \
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
#----powerline
COPY .bash_profile /etc/profile.d/bashrc.sh

RUN useradd -ms /bin/bash jovyan
###############################
#     VS Code Remote SSH      #
###############################
USER root

ENV NOTVISIBLE "in users profile"
RUN mkdir -p /vat/run/sshd 
RUN echo 'jovyan:esun@1313' | chpasswd 
# RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd 
RUN echo "export VISIBLE=now" >> /etc/profile 
RUN echo "PermitUserEnvironment yes" >> /etc/ssh/sshd_config
 
EXPOSE 22

#############################
#   Conda                   #
#############################
# USER $NB_UID

# COPY requirement-conda.txt $HOME
# RUN conda update --all
# RUN conda install --quiet --yes --file $HOME/requirement-conda.txt \
#  && conda clean -tipsy \
#  && fix-permissions $CONDA_DIR

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
# USER root

# COPY requirement-pip.txt $HOME

# RUN pip --default-timeout=10000 install --no-cache-dir -r $HOME/requirement-pip.txt

#############################
#   Jupyter Extension       #
#############################
# USER $NB_UID
# RUN jupyter nbextension install --py \
#         jupyter_dashboards \
#         --sys-prefix \
#  && jupyter nbextension enable --py jupyter_dashboards --sys-prefix \
#  && jupyter nbextension enable --py widgetsnbextension 
# RUN jupyter labextension install @jupyterlab/hub-extension \
#  && jupyter labextension install @jupyterlab/toc \
#  && jupyter labextension install @ryantam626/jupyterlab_sublime \
#  && jupyter labextension install jupyter-matplotlib \
#  && jupyter labextension install jupyter-cytoscape \
#  && jupyter labextension install jupyterlab-dash \
#  && jupyter labextension install jupyterlab-drawio \
#  && jupyter labextension install nbdime-jupyterlab 
# RUN jupyter lab clean -y \
#  && npm cache clean --force \ 
#  && rm -rf /home/$NB_USER/.cache/yarn \
#  && rm -rf /home/$NB_USER/.node-gyp 
# RUN fix-permissions $CONDA_DIR \
#  && fix-permissions /home/$NB_USER

# RUN pip install --no-cache-dir nbresuse \
#  && jupyter serverextension enable --py nbresuse \
#  && jupyter lab clean -y

#########################
#   Docker CLI          #
#########################
USER root
RUN apt install -y software-properties-common \
 && add-apt-repository ppa:libreoffice/ppa \
 && apt update
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add - \
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
##############################
# For PrimeHub Job Submission#
##############################
USER $NB_UID

ENV PATH $PATH:/opt/conda/bin
ENV PIP_CONFIG_FILE /etc/python/pip.config

COPY ai-cloud-pip.config /etc/python/pip.config
