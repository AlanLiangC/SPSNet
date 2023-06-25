#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4,5,6,7

set -x
NGPUS=$1
PY_ARGS=${@:2}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python -m torch.distributed.launch --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch ${PY_ARGS}

