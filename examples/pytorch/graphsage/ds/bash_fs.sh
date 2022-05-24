#!/bin/bash
bash run.sh 1 5 20 train_mt3 /data/ds/distdgl/ogbn-papers100M1/ogb-paper100M.json
sleep 20
sudo rm -rf /dev/shm/*
sleep 20
bash run.sh 2 10 20 train_mt3 /data/ds/distdgl/ogbn-papers100M2/ogb-paper100M.json
sleep 20
sudo rm -rf /dev/shm/*
sleep 20
bash run.sh 8 50 100 train_mt3 /data/ds/distdgl/ogbn-papers100M8/ogb-paper100M.json
sleep 20
sudo rm -rf /dev/shm/*
sleep 20
bash run.sh 1 10 0 train_mt3 /data/ds/friendster/friendster_1/friendster.json
sleep 20
sudo rm -rf /dev/shm/*
sleep 20
bash run.sh 2 20 0 train_mt3 /data/ds/friendster/friendster_2/friendster.json
sleep 20
sudo rm -rf /dev/shm/*
sleep 20
bash run.sh 4 40 0 train_mt3 /data/ds/friendster/friendster_4/friendster.json
sleep 20
sudo rm -rf /dev/shm/*
sleep 20
bash run.sh 8 50 20 train_mt3 /data/ds/friendster/friendster_8/friendster.json
sleep 20
sudo rm -rf /dev/shm/*
sleep 20
bash run.sh 1 10 0 train_seq /data/ds/friendster/friendster_1/friendster.json
sleep 20
sudo rm -rf /dev/shm/*
sleep 20
bash run.sh 2 20 0 train_seq /data/ds/friendster/friendster_2/friendster.json
sleep 20
sudo rm -rf /dev/shm/*
sleep 20
bash run.sh 4 40 0 train_seq /data/ds/friendster/friendster_4/friendster.json
sleep 20
sudo rm -rf /dev/shm/*
sleep 20
bash run.sh 8 50 20 train_seq /data/ds/friendster/friendster_8/friendster.json
sleep 20
sudo rm -rf /dev/shm/*
sleep 20