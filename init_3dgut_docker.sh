DATASET_DIR="/mnt/raid/dataset"
PROJECT_DIR="/home/admins/Documents/SIGGRAPH'25_GVR"

# docker build . -t 3dgrut

sudo docker remove -f 3dgut

echo "mount datasets: $DATASET_DIR --> 3dgut:/media"
echo "mount projects: $PROJECT_DIR --> gs1:/home"

xhost +local:root
docker run --name 3dgut -v --rm -it --gpus=all --net=host --ipc=host -v $PWD:/workspace \
    --volume="$DATASET_DIR:/media:rw" \
    --volume="$PROJECT_DIR:/home:rw" \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw --env=DISPLAY --env=QT_X11_NO_MINTSHM=1 \
    --runtime=nvidia -e DISPLAY 3dgrut
