SCENE_IDS="alameda berlin london nyc"

for SCENE_ID in $SCENE_IDS; do
    echo "Processing scene: $SCENE_ID"
    DATASET_DIR=data/zipnerf/fisheye/$SCENE_ID

    # python prepare_zipnerf_fish2equi.py \
    #     --path $DATASET_DIR \
    #     --src images_8 \
    #     --dst images_8_equidist

    python train.py --config-name apps/zipnerf_fisheye_3dgut.yaml path=$DATASET_DIR out_dir=runs/zipnerf_fisheye_3dgut experiment_name="$SCENE_ID" dataset.downsample_factor=8
done
