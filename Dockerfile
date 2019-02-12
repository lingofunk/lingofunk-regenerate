FROM python:3.6-jessie

COPY . /app
WORKDIR /app

RUN pip install -r ./requirements.txt
RUN python -m spacy download en
