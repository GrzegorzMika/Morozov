FROM python:3.7-buster

RUN apt-get update

COPY requirements.txt /home/Morozov/requirements.txt

RUN pip install -r /home/Morozov/requirements.txt

WORKDIR /home/Morozov

RUN mkdir /home/Morozov/cachedir

COPY . .

CMD python3 Beta.py && python3 -c "from utils import upload; upload()"