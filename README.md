## Intro

This project provides scripts inspired by [coral.ai](https://coral.ai) to setup and run on MacOS.

## Setup

**Clone and setup**

1. Clone this repository (`git clone https://github.com/Smiril/coral-ai-edge-tpu-video-watcher.git`)
2. Run script `./setup.sh` to create virtual env and install necessary Python dependencies. This may take several minutes to run.
3. 
	1) Run the commands from [coral.ai](https://coral.ai) to setup finaly.
	2) Run the commands from [axelera.ai](https://axelera.ai) to setup finaly.
4. ```make DOCKER_IMAGE=debian:bullseye DOCKER_CPUS="k8 armv7a aarch64" DOCKER_TARGETS=tests docker-build```
## Run the Code
```
python video.py --model_path model.tflite --label_path labels.txt --device cpu --threshold 0.5 --width 3840 --height 2160 --fps 30 --video-device 0
```

** • --device cpu / --device gpu → use model.tflite (float32) **

** • --device tpu → use model_int8_edgetpu.tflite (uint8 / int8, for once edgetpu_compiler-run) **

** • --device mps / --device npu → use model.mlpackage (for Apple MetalGPU AND Neural Engine) **

** • --device axelera → use model.onnx model.axm (for Axelera Neural Engine) **


## Links

```
https://github.com/Smiril/tflite-model-creator.git
```
