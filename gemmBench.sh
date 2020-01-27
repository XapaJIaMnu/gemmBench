#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    OMP_NUM_THREADS=1 taskset --cpu-list 0 ./bench $@
else
    OMP_NUM_THREADS=1 ./bench $@
fi
