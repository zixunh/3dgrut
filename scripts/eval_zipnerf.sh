SCENE_ID=alameda
OUTPUT_DIR=/workspace/runs/zipnerf_3dgut/"$SCENE_ID"/"$SCENE_ID"-2104_034135
DATASET_DIR=data/zipnerf/fisheye/$SCENE_ID
ITERS_NUM=30000

# evaluation
python metrics.py \
    -m $OUTPUT_DIR \
    --iters $ITERS_NUM \
    --custom_gt /home/ever_training/output/zipnerf/undistorted/"$SCENE_ID"_d4/test/ours_$ITERS_NUM/gt