#!/bin/bash
python partition_graph.py --dataset=friendster --num_parts=2 --output=/data/ds/friendster_2
sleep 20
python partition_graph.py --dataset=friendster --num_parts=4 --output=/data/ds/friendster_4
sleep 20
python partition_graph.py --dataset=friendster --num_parts=8 --output=/data/ds/friendster_8
sleep 20
python partition_graph.py --dataset=friendster --num_parts=1 --output=/data/ds/friendster_1