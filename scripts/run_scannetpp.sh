SCENE_ID=1f7cbbdde1
DATASET_DIR=/media/scannetpp/demo/$SCENE_ID/dslr

python prepare_scannetpp_fish2equi.py \
    --path $DATASET_DIR \
    --src resized_images \
    --dst images_equidist

python train.py --config-name apps/scannetpp_3dgut.yaml path=$DATASET_DIR out_dir=runs experiment_name="$SCENE_ID"_3dgut