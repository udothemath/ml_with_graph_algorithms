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

# How to test? 
# 1) Open container: 
# docker run -it -p 2222:22 -i 3e7ba83a6d95 bash
# service ssh start
# neo4j start
# cypher-shell
# user: neo4j pass: neo4j

# 2) On host:
# ssh -N -L 7474:127.0.0.1:7474 root@127.0.0.1 -p 2222
# Enter passwd: 'Docker!'
# ssh -N -L 7687:127.0.0.1:7687 root@127.0.0.1 -p 2222
# Enter passwd: 'Docker!'
# Open http://localhost:7474