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

#########################
#   UPDATE APT          #
#########################
USER root
RUN apt update \
 && apt install -y --no-install-recommends software-properties-common
RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common
RUN apt-get install -y --no-install-recommends \
        gnupg2 \
        lsb-core \
        gpg-agent

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
 && apt update
RUN apt install -y neo4j=1:4.4.6
RUN wget https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/4.4.0.6/apoc-4.4.0.6-all.jar \
 && cp apoc-4.4.0.6-all.jar /var/lib/neo4j/plugins/ \
 && chown neo4j:neo4j /var/lib/neo4j/plugins/apoc-4.4.0.6-all.jar

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
#   Conda                   #
#############################
USER $NB_UID
COPY requirement-conda.txt $HOME
RUN conda update --all --yes
RUN conda config --set channel_priority false
RUN conda install --yes --file $HOME/requirement-conda.txt \
 && conda clean -tipsy \
 && fix-permissions $CONDA_DIR


#############################
#       RedisInsight        #
#############################
USER root
RUN conda update --all --yes
RUN conda clean -tipsy
RUN fix-permissions $CONDA_DIR
RUN curl -sL https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - \
 && echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list \
 && apt update \
 && apt install -y yarn
RUN git clone https://github.com/RedisInsight/RedisInsight.git $HOME/Redisinsight
RUN cd $HOME/Redisinsight \
 && yarn install --network-timeout 100000
RUN cd $HOME/Redisinsight \
 && yarn --cwd redisinsight/api/

#############################
#         Ssh server        #
#############################
# https://linuxize.com/post/how-to-enable-ssh-on-ubuntu-18-04/
RUN apt update
RUN apt install -y openssh-server openssh-client
RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
RUN echo 'root:Docker!' | chpasswd
RUN mkdir /run/sshd
RUN service ssh start
EXPOSE 22

#############################
#    Test Neo4j Browser     #
#############################
# 1) Open container: 
# docker run -it -p 2222:22 -i d3b0bc7027ca bash
# service ssh start
# neo4j console

# 2) On host:
# ssh -N -L 7474:127.0.0.1:7474 root@127.0.0.1 -p 2222
# Enter passwd: 'Docker!'
# ssh -N -L 7687:127.0.0.1:7687 root@127.0.0.1 -p 2222
# Enter passwd: 'Docker!'
# Open http://localhost:7474

# NOTE: Success!

#############################
#    Test RedisInsight      #
#############################

# 1) Open container: 
# docker run -it -p 2222:22 -i e53f1cec4d43 bash
# service ssh start
# redis-stack-server &
# cd $HOME/RedisInsight
# yarn --cwd redisinsight/api/ start:dev &
# yarn start:web

# 2) On host:
# ssh -N -L 8080:127.0.0.1:8080 root@127.0.0.1 -p 2222
# Enter passwd: 'Docker!'
# ssh -N -L 5000:127.0.0.1:5000 root@127.0.0.1 -p 2222
# Enter passwd: 'Docker!'
# ssh -N -L 6379:127.0.0.1:6379 root@127.0.0.1 -p 2222
# Enter passwd: 'Docker!'
# Open http://localhost:8080

# NOTE: only redis-stack-server success