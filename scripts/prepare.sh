SCENE_ID=1f7cbbdde1
DATASET_DIR=/media/scannetpp/demo/$SCENE_ID/dslr

python prepare_scannetpp.py \
    --path $DATASET_DIR \
    --src resized_images \
    --dst image_undistorted_fisheye 

python prepare_scannetpp_fish2equi.py \
    --path $DATASET_DIR \
    --src resized_images \
    --dst images_equidist