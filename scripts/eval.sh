SCENE_ID=4ef75031e3
OUTPUT_DIR=/workspace/runs/"$SCENE_ID"_3dgut/dslr-1604_064548
DATASET_DIR=/media/scannetpp/demo/$SCENE_ID/dslr
ITERS_NUM=30000

# wrap back to origianal space
python prepare_scannetpp_equi2fish.py \
    --camera-path $DATASET_DIR/colmap/cameras_equidist.txt \
    --src $OUTPUT_DIR/ours_30000/renders \
    --dst $OUTPUT_DIR/ours_30000/renders_remap \

# evaluation
python metrics.py \
    -m $OUTPUT_DIR --use_remap \
    --iters $ITERS_NUM \
    --custom_gt /home/scannetpp_fs_gt/dslr/$SCENE_ID/test/ours_$ITERS_NUM/gt_remap