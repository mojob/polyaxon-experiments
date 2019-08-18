FROM python:3.7
MAINTAINER Brage Staven <bs@mojob.io>

COPY requirements.txt /tmp/
RUN cd /tmp && pip install --no-cache-dir -r requirements.txt

RUN mkdir -p polyaxon-experiments
WORKDIR polyaxon-experiments

COPY train.py /polyaxon-experiments
