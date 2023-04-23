## Intro

This project provides scripts inspired by [coral.ai](https://coral.ai) to setup and run on MacOS.

## Setup

**Clone and setup**

1. Clone this repository (`git clone https://github.com/Smiril/coral-ai-edge-tpu-video-watcher.git`)
2. Run script `./setup.sh` to create virtual env and install necessary Python dependencies. This may take several minutes to run.
3. Run the commands from [coral.ai](https://coral.ai) to setup finaly.
4. ```make DOCKER_IMAGE=debian:buster DOCKER_CPUS="k8 armv7a aarch64" DOCKER_TARGETS=tests docker-build```
## Run the Code
```
python video.py --model_path model.tflite --label_path labels.txt --device tpu --threshold 0.5 --width 640 --height 480 --video-device 0
```

## Links

```
https://github.com/Smiril/tflite-model-creator.git
```
