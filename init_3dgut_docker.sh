DATASET_DIR="/mnt/raid/dataset"

# docker build . -t 3dgrut

sudo docker remove -f 3dgut

echo "mount datasets: $DATASET_DIR --> 3dgut:/media"

xhost +local:root
docker run --name 3dgut -v --rm -it --gpus=all --net=host --ipc=host -v $PWD:/workspace \
    --volume="$DATASET_DIR:/media:rw" \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw --env=DISPLAY --env=QT_X11_NO_MINTSHM=1 \
    --runtime=nvidia -e DISPLAY 3dgrut
