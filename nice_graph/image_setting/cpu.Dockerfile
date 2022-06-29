FROM infuseai/docker-stacks:tensorflow-notebook-v2-4-1-dbdcead1
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
#   Update                  #
#############################
RUN conda install --update-specs --yes\
        mako==1.1.2

RUN pip install --upgrade \
        matplotlib==3.0.2 \
        urllib3

#############################
#   Jupyter Extension       #
#############################
USER $NB_UID

RUN jupyter nbextension install --py \
        jupyter_dashboards \
        --sys-prefix \
 && jupyter nbextension enable --py jupyter_dashboards --sys-prefix \
 && jupyter nbextension enable --py widgetsnbextension

RUN jupyter labextension install \
        @bokeh/jupyter_bokeh \
        @jupyterlab/hub-extension \
        @jupyterlab/toc \
        @ryantam626/jupyterlab_sublime \
        jupyter-matplotlib \
        jupyterlab_filetree \
        jupyterlab_tensorboard \
        jupyterlab-dash \
        jupyterlab-drawio \
        nbdime-jupyterlab \
        --no-build \
 && jupyter lab build -y \
 && jupyter lab clean -y \
 && npm cache clean --force \ 
 && rm -rf /home/$NB_USER/.cache/yarn \
 && rm -rf /home/$NB_USER/.node-gyp \
 && fix-permissions $CONDA_DIR \
 && fix-permissions /home/$NB_USER

RUN pip install --no-cache-dir nbresuse \
 && jupyter serverextension enable --py nbresuse \
 && jupyter lab clean -y

#############################
#   Julia                   #
#############################
USER root

RUN wget --directory-prefix=/usr/local/lib/ https://julialang-s3.julialang.org/bin/linux/x64/1.1/julia-1.1.0-linux-x86_64.tar.gz \
 && tar -xvf /usr/local/lib/julia-1.1.0-linux-x86_64.tar.gz -C /usr/local/lib/ \
 && rm /usr/local/lib/julia-1.1.0-linux-x86_64.tar.gz \
 && chown -R jovyan:users /usr/local/lib/julia-1.1.0/

ENV PATH "/usr/local/lib/julia-1.1.0/bin:$PATH"
ENV JULIA_DEPOT_PATH "/usr/local/lib/julia-1.1.0/"
ENV JUPYTER "/opt/conda/bin/jupyter-labhub"

RUN julia -e 'using Pkg; Pkg.add("PyPlot"); Pkg.build("PyPlot"); \
        Pkg.add("IJulia"); Pkg.build("IJulia"); \
        Pkg.add("NetCDF"); Pkg.build("NetCDF"); \
        Pkg.add("MAT"); Pkg.build("MAT")' \
 && chown -R jovyan:users /usr/local/lib/julia-1.1.0/

#############################
#   Tesseract Package       #
#############################
RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common \
 && echo "deb http://ppa.launchpad.net/alex-p/tesseract-ocr/ubuntu bionic main" >> /etc/apt/source.list \
 && echo "deb-src http://ppa.launchpad.net/alex-p/tesseract-ocr/ubuntu bionic main" >> /etc/apt/source.list \
 && add-apt-repository --yes ppa:alex-p/tesseract-ocr \
 && apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-chi-tra

#########################
#   Docker CLI          #
#########################
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - \
 && add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable" \
 && apt-get update \
 && apt-get install -y --no-install-recommends docker-ce-cli
 
#############################
#       RPA Package         #
#############################
RUN wget https://chromedriver.storage.googleapis.com/87.0.4280.88/chromedriver_linux64.zip \
 && unzip chromedriver_linux64.zip \
 && chmod +x chromedriver \
 && mv chromedriver /usr/bin/ \
 && rm chromedriver_linux64.zip

##############################
# For PrimeHub Job Submission#
##############################
USER $NB_UID

ENV PATH $PATH:/opt/conda/bin
ENV PIP_CONFIG_FILE /etc/python/pip.config

COPY ai-cloud-pip.config /etc/python/pip.config
