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
