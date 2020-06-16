FROM python:3.7-buster

RUN apt-get update

COPY requirements.txt /home/Morozov/requirements.txt

RUN pip install -r /home/Morozov/requirements.txt

RUN pip install dask distributed --upgrade

WORKDIR /home/Morozov

RUN mkdir /home/Morozov/cachedir

RUN mkdir /home/Morozov/output

COPY . .

CMD python3 Simulations.py