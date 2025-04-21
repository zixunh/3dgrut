SCENE_IDS="berlin london nyc"

for SCENE_ID in $SCENE_IDS; do
    echo "Processing scene: $SCENE_ID"
    DATASET_DIR=data/zipnerf/fisheye/$SCENE_ID

    python train.py --config-name apps/zipnerf_3dgut.yaml path=$DATASET_DIR out_dir=runs/zipnerf_3dgut experiment_name="$SCENE_ID" dataset.downsample_factor=4
done
