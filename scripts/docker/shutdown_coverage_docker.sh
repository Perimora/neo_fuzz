#!/bin/bash

# get container ID from cmd line argument
C_ID="$1"

# check if container exists and delete it by ID
if [ "$(docker ps -aq -f id=$C_ID)" ]; then
    docker rm -f $C_ID
fi
