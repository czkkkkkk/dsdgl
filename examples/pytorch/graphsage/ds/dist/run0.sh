#!/bin/bash
n_gpu=$1
#datadir=/data/ds/friendster/friendster_${n_gpu}/friendster.json
#datadir=/data/ds/test/fake2/fake.json
# datadir=/data/ds/test/with_outedge_ogb-product${n_gpu}/ogb-product.json
datadir=$5
datadir=./data/reddit${n_gpu}/reddit.json
# datadir=/data/ds/metis_ogbn-papers100M${n_gpu}/ogbn-papers100M.json
datadir=/data/ds/distdgl/ogbn-papers100M${n_gpu}/ogb-paper100M.json
datadir=../data/ogb-product${n_gpu}/ogb-product.json
num_hidden=16
num_hidden=256
fanout=10,25
fanout=2,2
fanout=5,10,15
epochs=20
# export DGL_DS_MASTER_PORT=12338
export DGL_DS_N_BLOCK=32
export DGL_DS_USE_NCCL=1
export DGL_DS_MASTER_PORT=12210
export DGL_DS_COMM_PORT=17211
cache_ratio=$2
cache_ratio=50
graph_cache_ratio=100
feat_cache_gb=3
# feat_cache_gb=$2
graph_cache_gb=3
# graph_cache_gb=$3
feat_mode=AllCache
feat_mode=DistPartitionCache
batch_size=1024

node_rank=0
# DGL_DS_USE_NCCL=1 DGL_DS_N_BLOCK=32 python train.py --part_config=${datadir} --n_ranks=${n_gpu} --fan_out=${fanout} --num_hidden=${num_hidden} 
#DGL_DS_USE_NCCL=1 torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="172.31.42.109" --master_port=1258 \
DGL_DS_USE_NCCL=1 DGL_DS_ROOT_ADDR=172.31.42.109 DGL_DS_MY_ADDR=172.31.42.109 \
                python dist_dsp.py --part_config=${datadir} \
                --nnodes=2 \
                --nproc_per_node=${n_gpu} \
                --node_rank=${node_rank} \
                --fan_out=${fanout} --num_hidden=${num_hidden} \
                --num_epochs=${epochs} --cache_ratio=${cache_ratio} --feat_mode=${feat_mode} --graph_cache_ratio=${graph_cache_ratio}  \
               --batch_size=${batch_size} \
               --graph_cache_gb=${graph_cache_gb} \
               --feat_cache_gb=${feat_cache_gb}
