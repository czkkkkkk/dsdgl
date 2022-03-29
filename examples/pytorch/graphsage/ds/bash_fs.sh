#!/bin/bash
n_parts=1
python partition_graph.py --dataset=friendster --num_parts=${n_parts} --output=/data/ds/friendster_${n_parts}
n_parts=4
python partition_graph.py --dataset=friendster --num_parts=${n_parts} --output=/data/ds/friendster_${n_parts}
python partition_graph.py --dataset=friendster --num_parts=${n_parts} --output=/data/ds/friendster_${n_parts}
n_parts=8
python partition_graph.py --dataset=friendster --num_parts=${n_parts} --output=/data/ds/friendster_${n_parts}
python partition_graph.py --dataset=friendster --num_parts=${n_parts} --output=/data/ds/friendster_${n_parts}
# sleep 20
# python partition_graph.py --dataset=friendster --num_parts=4 --output=/data/ds/friendster_4 --undirected=False
# sleep 20
# python partition_graph.py --dataset=friendster --num_parts=8 --output=/data/ds/friendster_8 --undirected=False
# sleep 20
# python partition_graph.py --dataset=friendster --num_parts=1 --output=/data/ds/friendster_1 --undirected=False
