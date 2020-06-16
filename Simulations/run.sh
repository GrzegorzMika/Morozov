#!/bin/bash
# Build image and run it
docker volume create output
docker volume create cachedir
docker build -t simulations -f Dockerfile .
docker run -d -v cachedir:/home/Morozov/cachedir \
              -v output:/home/Morozov/output \
              --security-opt seccomp=unconfined \
              simulations:latest