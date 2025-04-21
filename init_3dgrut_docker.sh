DATASET_DIR="/mnt/data_ssd_4tb/Datasets/"
PROJECT_DIR="/home/yuliang/Projects/"

# docker build . -t 3dgrut

sudo docker remove -f 3dgrut

echo "mount datasets: $DATASET_DIR --> 3dgrut:/media"
echo "mount projects: $PROJECT_DIR --> 3dgrut:/home"

xhost +local:root
docker run --name 3dgrut -v --rm -it --gpus=all --net=host --ipc=host -v $PWD:/workspace \
    --volume="$DATASET_DIR:/media:rw" \
    --volume="$PROJECT_DIR:/home:rw" \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw --env=DISPLAY --env=QT_X11_NO_MINTSHM=1 \
    --runtime=nvidia -e DISPLAY 3dgrut
