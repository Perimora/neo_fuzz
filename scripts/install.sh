#!/bin/bash

sudo apt update

# install docker
sudo apt-get install -y docker.io

# install lua and teal
sudo apt install -y lua5.3 liblua5.3-dev luarocks

sudo luarocks install tl

tl --version
