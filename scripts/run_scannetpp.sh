SCENE_ID=0a5c013435
DATASET_DIR=/media/scannetpp/demo/$SCENE_ID/dslr

# python prepare_scannetpp.py \
#     --path $DATASET_DIR \
#     --src resized_images \
#     --dst image_undistorted_fisheye 

# python prepare_scannetpp_fish2equi.py \
#     --path /media/scannetpp/demo/0a5c013435/dslr/ \
#     --src resized_images \
#     --dst images_equidist

python train.py --config-name apps/scannetpp_3dgut.yaml path=$DATASET_DIR out_dir=runs experiment_name="$SCENE_ID"_3dgut


# # render
# python render.py \
#     -m $OUTPUT_PATH \
#     -s $DATASET_PATH \
#     --iteration 30000 \
#     --camera_model FISHEYE \
#     -r 1 \
#     --skip_train

# # wrap back to origianal space
# python prepare_scannetpp_equi2fish.py \
#     --camera-path $DATASET_PATH/colmap/cameras_equidist.txt \
#     --src $OUTPUT_PATH/test/ours_30000/gt \
#     --dst $OUTPUT_PATH/test/ours_30000/gt_remap \

# python prepare_scannetpp_equi2fish.py \
#     --camera-path $DATASET_PATH/colmap/cameras_equidist.txt \
#     --src $OUTPUT_PATH/test/ours_30000/renders \
#     --dst $OUTPUT_PATH/test/ours_30000/renders_remap \

# # evaluation
# python metrics.py \
#     -m $OUTPUT_PATH \
#     --use_remap