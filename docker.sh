#!/usr/bin/env bash

DOCKER_ADDRESS=registry.aibee.cn/product_analysis/torch:1.8.2

sudo docker pull $DOCKER_ADDRESS

sudo docker run --shm-size=12gb -it -d \
    --name iv_tp \
    --network=host \
    -e COLUMNS=`tput cols` \
    -e LINES=`tput lines` \
    -v /etc/localtime:/etc/localtime:ro \
    -v /ssd:/ssd \
    -v /mnt:/mnt \
    -v /training:/training \
    -v /face:/face \
    -v $PWD/:/workspace \
    -p 12355:12355 \
    $DOCKER_ADDRESS \
    bash