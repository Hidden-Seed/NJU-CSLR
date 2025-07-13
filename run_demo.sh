#!/bin/bash

DEMO_INDEX=0

while getopts "n:" opt; do
  case $opt in
    n) DEMO_INDEX=$OPTARG ;;  # -n 参数用于指定 demo 编号
    *) echo "Usage: $0 [-n demo_index]" ;;
  esac
done

echo "Running demo $DEMO_INDEX..."
python prediction.py --prediction_type local_$DEMO_INDEX

# Demo_index
# 0 - Simple prediction
# 1 - Accuracy test
