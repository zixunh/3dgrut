SCENE_IDS="4ef75031e3 1d003b07bd"

for SCENE_ID in $SCENE_IDS; do
    echo "Processing scene: $SCENE_ID"
    DATASET_DIR=/media/scannetpp/demo/$SCENE_ID/dslr

    python prepare_scannetpp_fish2equi.py \
        --path $DATASET_DIR \
        --src resized_images \
        --dst images_equidist

    python train.py --config-name apps/scannetpp_3dgut.yaml path=$DATASET_DIR out_dir=runs experiment_name="$SCENE_ID"_3dgut
done
