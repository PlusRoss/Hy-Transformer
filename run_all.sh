#!/bin/bash

./run_batch_wd50k.sh 0 wd50k statements StarE hy-transformer_mask True &
./run_batch_jf17k.sh 1 jf17k statements StarE hy-transformer_mask True &
./run_batch_wikipeople.sh 2 wikipeople statements StarE hy-transformer_mask True &
