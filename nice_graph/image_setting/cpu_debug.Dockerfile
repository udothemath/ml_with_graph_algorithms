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
RUN conda config --set channel_priority false
RUN conda install --yes --file $HOME/requirement-conda.txt \
 && conda clean -tipsy \
 && fix-permissions $CONDA_DIR


#############################
#       RedisInsight        #
#############################
USER root
RUN curl -sL https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - \
 && echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list \
 && apt update \
 && apt install -y yarn
RUN conda update -c bitsort nodejs
RUN conda clean -tipsy
RUN fix-permissions $CONDA_DIR
RUN git clone https://github.com/RedisInsight/RedisInsight.git 
RUN yarn --cwd $HOME/RedisInsight/redisinsight/api/ 
RUN cd $HOME/RedisInsight \
 && npm install --save-dev node-gyp -g \
 && npm install --save-dev webpack webpack-cli webpack-dev-server -g \
 && npm install --save-dev cross-env -get 
RUN cd $HOME/RedisInsight \
 && npm install --save-dev @babel/core@^7.0.0-0 -get \
 && npm install --save-dev @babel/register -get \
 && npm install --save-dev @babel/plugin-transform-runtime -get \
 && npm install --save-dev react@^15.0.0 -get \
 && npm install --save-dev @types/react@^15.0.0 -get \
 && npm install --save-dev react-dom@^15.0.0 -get \
 && npm install --save-dev react-hot-loader -get \
 && npm install --save-dev @elastic/datemath@^5.0.3 -get \
 && npm install --save-dev @elastic/eui@34.6.0 -get \
 && npm install --save-dev @reduxjs/toolkit@^1.6.2 -get \
 && npm install --save-dev axios@^0.25.0 -get \
 && npm install --save-dev classnames@^2.3.1 -get \
 && npm install --save-dev connection-string@^4.3.2 -get \
 && npm install --save-dev date-fns@^2.16.1 -get \
 && npm install --save-dev detect-port@^1.3.0 -get \
 && npm install --save-dev electron-context-menu@^3.1.0 -get \
 && npm install --save-dev electron-log@^4.2.4 -get \
 && npm install --save-dev electron-store@^8.0.0 -get \
 && npm install --save-dev electron-updater@4.6.5 -get \
 && npm install --save-dev formik@^2.2.9 -get \
 && npm install --save-dev html-entities@^2.3.2 -get \
 && npm install --save-dev html-react-parser@^1.2.4 -get \
 && npm install --save-dev jsonpath@^1.1.1 -get \
 && npm install --save-dev lodash@^4.17.21 -get \
 && npm install --save-dev react-contenteditable@^3.3.5 -get \
 && npm install --save-dev react-hotkeys-hook@^3.3.1 -get \
 && npm install --save-dev react-monaco-editor@^0.44.0 -get \
 && npm install --save-dev react-redux@^7.2.2 -get \
 && npm install --save-dev react-rnd@^10.3.5 -get \
 && npm install --save-dev react-router-dom@^5.2.0 -get \
 && npm install --save-dev react-virtualized@^9.22.2 -get \
 && npm install --save-dev react-virtualized-auto-sizer@^1.0.6 -get \
 && npm install --save-dev react-vtree@^3.0.0-beta.3 -get \
 && npm install --save-dev react-window@^1.8.6 -get \
 && npm install --save-dev react-window-infinite-loader@^1.0.8 -get \
 && npm install --save-dev rehype-stringify@^9.0.2 -get \
 && npm install --save-dev remark-gfm@^3.0.1 -get \
 && npm install --save-dev remark-parse@^10.0.1 -get \
 && npm install --save-dev remark-rehype@^10.0.1 -get \
 && npm install --save-dev socket.io-client@^4.4.0 -get \
 && npm install --save-dev unified@^10.1.1 -get \
 && npm install --save-dev unist-util-visit@^4.1.0 -get \
 && npm install --save-dev url-parse@^1.5.10 -get \
 && npm install --save-dev uuid@^8.3.2 -get \
 && npm install --save-dev @babel/plugin-proposal-class-properties@^7.12.1 -get \
 && npm install --save-dev @babel/plugin-proposal-decorators@^7.12.1 -get \
 && npm install --save-dev @babel/plugin-proposal-do-expressions@^7.12.1 -get \
 && npm install --save-dev @babel/plugin-proposal-export-default-from@^7.12.1 -get \
 && npm install --save-dev @babel/plugin-proposal-export-namespace-from@^7.12.1 -get \
 && npm install --save-dev @babel/plugin-proposal-function-bind@^7.12.1 -get \
 && npm install --save-dev @babel/plugin-proposal-function-sent@^7.12.1 -get \
 && npm install --save-dev @babel/plugin-proposal-json-strings@^7.12.1 -get \
 && npm install --save-dev @babel/plugin-proposal-logical-assignment-operators@^7.12.1 -get \
 && npm install --save-dev @babel/plugin-proposal-nullish-coalescing-operator@^7.12.1 -get \
 && npm install --save-dev @babel/plugin-proposal-optional-chaining@^7.12.7 -get \
 && npm install --save-dev @babel/plugin-proposal-pipeline-operator@^7.12.1 -get \
 && npm install --save-dev @babel/plugin-proposal-throw-expressions@^7.12.1 -get \
 && npm install --save-dev @babel/plugin-syntax-dynamic-import@^7.8.3 -get \
 && npm install --save-dev @babel/plugin-syntax-import-meta@^7.10.4 -get \
 && npm install --save-dev @babel/plugin-transform-react-constant-elements@^7.12.1 -get \
 && npm install --save-dev @babel/plugin-transform-react-inline-elements@^7.12.1 -get \
 && npm install --save-dev @babel/preset-env@^7.12.7 -get \
 && npm install --save-dev @babel/preset-react@^7.12.7 -get \
 && npm install --save-dev @babel/preset-typescript@^7.12.7 -get \
 && npm install --save-dev @babel/preset-typescript@^7.12.7 -get \
 && npm install --save-dev @nestjs/cli@^7.0.0 -get \
 && npm install --save-dev @nestjs/schematics@^7.0.0 -get \
 && npm install --save-dev @nestjs/testing@^7.0.0 -get \
 && npm install --save-dev @pmmmwh/react-refresh-webpack-plugin@^0.4.3 -get \
 && npm install --save-dev @svgr/webpack@^6.2.1 -get \
 && npm install --save-dev @teamsupercell/typings-for-css-modules-loader@^2.4.0 -get \
 && npm install --save-dev @testing-library/jest-dom@^5.11.6 -get \
 && npm install --save-dev @testing-library/react@^11.2.2 -get \
 && npm install --save-dev @types/axios@^0.14.0 -get \
 && npm install --save-dev @types/classnames@^2.2.11 -get \
 && npm install --save-dev @types/date-fns@^2.6.0 -get \
 && npm install --save-dev @types/detect-port@^1.3.0 -get \
 && npm install --save-dev @types/electron-store@^3.2.0 -get \
 && npm install --save-dev @types/express@^4.17.3 -get \
 && npm install --save-dev @types/html-entities@^1.3.4 -get \
 && npm install --save-dev @types/ioredis@^4.26.0 -get \
 && npm install --save-dev @types/is-glob@^4.0.2 -get \
 && npm install --save-dev @types/jest@^26.0.15 -get \
 && npm install --save-dev @types/lodash@^4.14.171 -get \
 && npm install --save-dev @types/node@14.14.10 -get \
 && npm install --save-dev @types/react-dom@^17.0.0 -get 
RUN cd $HOME/RedisInsight \
 && npm install --save-dev @types/react-monaco-editor@^0.16.0 -get \
 && npm install --save-dev @types/react-redux@^7.1.12 -get \
 && npm install --save-dev @types/react-router-dom@^5.1.6 -get \
 && npm install --save-dev @types/react-virtualized@^9.21.10 -get \
 && npm install --save-dev @types/react-window-infinite-loader@^1.0.6 -get \
 && npm install --save-dev @types/redux-mock-store@^1.0.2 -get \
 && npm install --save-dev @types/segment-analytics@^0.0.34 -get \
 && npm install --save-dev @types/supertest@^2.0.8 -get \
 && npm install --save-dev @types/uuid@^8.3.4 -get \
 && npm install --save-dev @types/webpack-env@^1.15.2 -get \
 && npm install --save-dev @typescript-eslint/eslint-plugin@^4.8.1 -get \
 && npm install --save-dev @typescript-eslint/parser@^4.8.1 -get \
 && npm install --save-dev babel-eslint@^10.1.0 -get \
 && npm install --save-dev babel-jest@^26.1.0 -get \
 && npm install --save-dev babel-loader@^8.2.2 -get \
 && npm install --save-dev babel-plugin-dev-expression@^0.2.2 -get \
 && npm install --save-dev babel-plugin-parameter-decorator@^1.0.16 -get \
 && npm install --save-dev babel-plugin-transform-react-remove-prop-types@^0.4.24 -get \
 && npm install --save-dev cache-loader@^4.1.0 -get \
 && npm install --save-dev concurrently@^5.3.0 -get \
 && npm install --save-dev core-js@^3.6.5 -get \
 && npm install --save-dev css-loader@^5.0.1 -get \
 && npm install --save-dev css-minimizer-webpack-plugin@^1.2.0 -get
RUN cd $HOME/RedisInsight \
 && npm install --save-dev eslint@^7.5.0 -get \
 && npm install --save-dev eslint-config-airbnb@^18.2.1 -get \
 && npm install --save-dev eslint-config-airbnb-typescript@^12.0.0 -get \
 && npm install --save-dev eslint-import-resolver-webpack@0.13.0 -get \
 && npm install --save-dev eslint-plugin-compat@^3.8.0 -get \
 && npm install --save-dev eslint-plugin-import@^2.22.0 -get \
 && npm install --save-dev eslint-plugin-jest@^24.1.3 -get \
 && npm install --save-dev eslint-plugin-jsx-a11y@6.4.1 -get \
 && npm install --save-dev eslint-plugin-promise@^4.2.1 -get \
 && npm install --save-dev eslint-plugin-react@^7.20.6 -get \
 && npm install --save-dev eslint-plugin-react-hooks@^4.0.8 -get \
 && npm install --save-dev eslint-plugin-sonarjs@^0.10.0 -get \
 && npm install --save-dev file-loader@^6.0.0 -get \
 && npm install --save-dev html-webpack-plugin@^4.5.0 -get \
 && npm install --save-dev husky@^4.2.5 -get \
 && npm install --save-dev identity-obj-proxy@^3.0.0 -get \
 && npm install --save-dev ioredis-mock@^5.5.4 -get \
 && npm install --save-dev ip@^1.1.8 -get \
 && npm install --save-dev jest@^26.1.0 -get \
 && npm install --save-dev jest-when@^3.2.1 -get \
 && npm install --save-dev lint-staged@^10.2.11 -get \
 && npm install --save-dev mini-css-extract-plugin@^1.3.1 -get \
 && npm install --save-dev monaco-editor-webpack-plugin@^6.0.0 -get \
 && npm install --save-dev opencollective-postinstall@^2.0.3 -get \
 && npm install --save-dev react-refresh@^0.9.0 -get \
 && npm install --save-dev redux-mock-store@^1.5.4 -get \
 && npm install --save-dev regenerator-runtime@^0.13.5 -get \
 && npm install --save-dev rimraf@^3.0.2 -get \
 && npm install --save-dev sass-loader@^10.1.0 -get \
 && npm install --save-dev skip-postinstall@^1.0.0 -get \
 && npm install --save-dev socket.io-mock@^1.3.2 -get \
 && npm install --save-dev style-loader@^2.0.0 -get \
 && npm install --save-dev supertest@^4.0.2 -get \
 && npm install --save-dev terser-webpack-plugin@^5.0.3 -get \
 && npm install --save-dev ts-jest@26.1.0 -get \
 && npm install --save-dev ts-loader@^6.2.1 -get \
 && npm install --save-dev ts-mockito@^2.6.1 -get \
 && npm install --save-dev ts-node@^8.6.2 -get \
 && npm install --save-dev tsconfig-paths@^3.9.0 -get \
 && npm install --save-dev tsconfig-paths-webpack-plugin@^3.3.0 -get \
 && npm install --save-dev typescript@^4.0.5 -get \
 && npm install --save-dev url-loader@^4.1.0 -get \
 && npm install --save-dev webpack@^5.5.1 -get \
 && npm install --save-dev webpack-bundle-analyzer@^4.1.0 -get \
 && npm install --save-dev webpack-cli@^4.3.0 -get \
 && npm install --save-dev webpack-dev-server@^3.11.0 -get \
 && npm install --save-dev webpack-merge@^5.4.0 -get \
 && npm install --save-dev yarn-deduplicate@^3.1.0 -get \
 && npm install --save-dev electron-builder@^22.14.13 -get \
 && npm install --save-dev electron-builder-notarize@^1.2.0 -get \
 && npm install --save-dev electron-debug@^3.1.0 -get \
 && npm install --save-dev electron-devtools-installer@^3.2.0 -get
# Failing Commands:
# RUN cd $HOME/RedisInsight \
#  && npm install --save-dev electron-rebuild@^2.3.2 -get
# RUN cd $HOME/RedisInsight \
#  && npm install --save-dev electron@^16.0.8 -get
# RUN cd $HOME/RedisInsight \
#  && npm install --save-dev react-jsx-parser@^1.28.4 -get
# RUN cd $HOME/RedisInsight \
#  && npm install --save-dev -sass@^5.0.0 -get
# RUN cd $HOME/RedisInsight \
#  && npm install --save-dev --production; exit 0
# RUN cd $HOME/RedisInsight \
#  && npm audit fix
# RUN cd $HOME/RedisInsight \
#  && yarn add web 
# RUN cd $HOME/RedisInsight \
#  && yarn add webpack-cli 
RUN cd $HOME

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