SCENE_ID=nyc
OUTPUT_DIR=/workspace/runs/zipnerf_fisheye_3dgut/"$SCENE_ID"/"$SCENE_ID"-2004_235332
DATASET_DIR=data/zipnerf/fisheye/$SCENE_ID
ITERS_NUM=30000

# wrap back to origianal space
python prepare_zipnerf_equi2fish.py \
    --camera-path $DATASET_DIR/sparse/0/cameras.bin \
    --src $OUTPUT_DIR/ours_30000/renders \
    --dst $OUTPUT_DIR/ours_30000/renders_remap \
    -r 8 \

# evaluation
python metrics.py \
    -m $OUTPUT_DIR --use_remap \
    --iters $ITERS_NUM \
    --custom_gt /home/Fisheye-GS/output/zipnerf/fisheye/$SCENE_ID/test/ours_$ITERS_NUM/gt_remap