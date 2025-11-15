#!/bin/sh

# stylegan2_ffhq1024 --classifier_model_path=classifier/ft_model_epoch_8.pth -L 6-7 -N 5 -K 7

MODEL_NAME=stylegan2_ffhq1024
classifier_model_path=classifier/ft_model_epoch_8.pth
# LAYER_IDX=0,1,2,3,4,5,6,7,8
NUM_SAMPLES=8
NUM_SEMANTICS=9
for LAYER_IDX in 0 1 2 3 4 5 6 7 8 9
do
  python main_correlation_estimator.py ${MODEL_NAME} \
    --classifier_model_path=$classifier_model_path \
    -L ${LAYER_IDX}-${LAYER_IDX} \
    -N ${NUM_SAMPLES} \
    -K ${NUM_SEMANTICS} \
    --thres_cc=0.45
done

