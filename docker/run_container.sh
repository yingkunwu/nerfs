WORKSPACE_DIR="./"
IMAGE_NAME=colmap
IMAGE_VERSION=latest

docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --network="host" \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v $WORKSPACE_DIR:/workspace \
    $IMAGE_NAME:$IMAGE_VERSION
