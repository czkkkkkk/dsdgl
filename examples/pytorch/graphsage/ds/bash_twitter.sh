#!/bin/bash
python partition_graph.py --dataset=twitter --num_parts=2 --output=/data/ds/twitter_2
sleep 20
python partition_graph.py --dataset=twitter --num_parts=4 --output=/data/ds/twitter_4
sleep 20
python partition_graph.py --dataset=twitter --num_parts=8 --output=/data/ds/twitter_8
sleep 20
python partition_graph.py --dataset=twitter --num_parts=1 --output=/data/ds/twitter_1