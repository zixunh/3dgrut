SCENE_ID=nyc
OUTPUT_DIR=/workspace/runs/zipnerf_3dgut/"$SCENE_ID"/"$SCENE_ID"-2104_050242
DATASET_DIR=data/zipnerf/fisheye/$SCENE_ID
ITERS_NUM=30000

# render cross camera
python render.py \
    --checkpoint $OUTPUT_DIR/ckpt_last.pt --out-dir $OUTPUT_DIR --cross-camera --downsample-factor 8\

# wrap back to origianal space
python prepare_zipnerf_equi2fish.py \
    --camera-path $DATASET_DIR/sparse/0/cameras.bin \
    --src $OUTPUT_DIR/ours_30000/renders_cross_camera \
    --dst $OUTPUT_DIR/ours_30000/renders_cross_camera_remap \
    -r 8 \

# wrap back to origianal space
python prepare_zipnerf_equi2fish.py \
    --camera-path $DATASET_DIR/sparse/0/cameras.bin \
    --src $OUTPUT_DIR/ours_30000/gt_cross_camera \
    --dst $OUTPUT_DIR/ours_30000/gt_cross_camera_remap \
    -r 8 \

# evaluation
python metrics.py \
    -m $OUTPUT_DIR --use_remap --cross_camera \
    --iters $ITERS_NUM \