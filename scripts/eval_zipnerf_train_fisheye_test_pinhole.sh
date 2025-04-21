SCENE_ID=nyc
OUTPUT_DIR=/workspace/runs/zipnerf_fisheye_3dgut/"$SCENE_ID"/"$SCENE_ID"-2004_235332
ITERS_NUM=30000

# render cross camera
python render.py \
    --checkpoint $OUTPUT_DIR/ckpt_last.pt --out-dir $OUTPUT_DIR --cross-camera --downsample-factor 4\

# evaluation
python metrics.py \
    -m $OUTPUT_DIR --cross_camera \
    --iters $ITERS_NUM \