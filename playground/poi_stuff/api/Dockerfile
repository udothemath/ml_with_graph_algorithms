ARG IMAGE_REGISTRY=""
FROM ${IMAGE_REGISTRY}/library/fastapi-py37:v0.1.4
# FROM ${IMAGE_REGISTRY}/up0125/fastapi:v1.0.0-py37-debug
# 普通api 使用 v0.0.11/v1.0.0-py37版本

WORKDIR /app

ARG PROJECT_NAME=""
ARG METAKEY1=""
ARG METAKEY2=""
ARG MLAAS_ENV=""
ARG MODEL_VERSION=""
ARG INDEX_URL=""
ARG TRUSTED_HOST=""

ENV LANG C.UTF-8
ENV PROJECT_NAME=${PROJECT_NAME}
ENV METAKEY1=${METAKEY1}
ENV METAKEY2=${METAKEY2}
ENV MLAAS_ENV=${MLAAS_ENV}
ENV MODEL_VERSION=${MODEL_VERSION}


COPY ./requirements.txt .
RUN pip3 install -U pip==21.2.1 -i ${INDEX_URL} --trusted-host=${TRUSTED_HOST}
RUN pip3 install -r requirements.txt -i ${INDEX_URL} --trusted-host=${TRUSTED_HOST}

COPY . .

USER root
RUN chown -R 1001:0 /app
USER 1001

EXPOSE 8000
ENTRYPOINT ["gunicorn"]
CMD ["app:app", "-c", "gunicorn_conf.py"]
