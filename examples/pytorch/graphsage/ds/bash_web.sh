#!/bin/bash
python partition_graph.py --dataset=webuk --num_parts=1 --output=/data/ds/web_uk_1
sleep 20
python partition_graph.py --dataset=webuk --num_parts=2 --output=/data/ds/web_uk_2
sleep 20
python partition_graph.py --dataset=webuk --num_parts=4 --output=/data/ds/web_uk_4
sleep 20
python partition_graph.py --dataset=webuk --num_parts=8 --output=/data/ds/web_uk_8