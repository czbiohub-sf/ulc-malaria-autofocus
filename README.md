# Single-Shot Auto Focus (SSAF) for the Remoscope
This is work done by the Bioengineering team at the CZ Biohub SF, a non-profit research institution.
![CZ Biohub SF logo](images/CZ-Biohub-SF-Color-RGB.png)

This repository houses a simple CNN used for running single-shot autofocus on the Remoscope (preprint here: https://www.medrxiv.org/content/10.1101/2024.11.12.24317184v1).
On Remoscope, we flow whole blood through a flow cell whose thickness is such that the red blood cells flow in a monolayer. We want to be able to image continuously without needing to pause
and acquire Z-stacks to re-focus (in the case of focal drift caused by mechanical vibrations or thermal effects).

To keep the sample in focus throughout the run, we have trained a CNN to recognize focus offsets based on the images (there is an asymmetry in the focus stack due to non-idealities in the optics which
causes the images to look slightly different when above or below focus). The model outputs the number of steps away from focus and the direction (i.e +3 steps, -5 steps, etc.).

## Installation

These instructions are for installation onto the BRUNO HPC at the Chan Zuckerberg Biohub San Francisco.

Create (or activate) a conda environment if you have not already.

Install PyTorch and Torchvision from [here](https://pytorch.org/get-started/locally/), selecting the installation settings for

- Stable
- Linux
- Conda
- Python
- Cuda (most recent, or the Cuda version that the HPC uses)

```console
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```

For exporting models, we need `onnx`:

```console
conda install -c conda-forge onnx
```

And for converting to the intermediate representation,

```console
pip3 install openvino-dev
```

If you find that you need a package that is not included here, please create an issue!


## Training

The file `ssaf_training.sh` is the batch job file. Run

```console
sbatch ssaf_training.sh
```


## Exporting Models

We need to convert your `.pth` file to `.onnx` and then to the Intel intermediate representation (a `.xml` and a `.bin` file). To export your nmodel to `.onnx`, run

```console
./to_onnx.py <PATH_TO_YOUR_pth>
```

which should create a `.onnx` file, a `.xml` file, and a `.bin` file. The `.bin` and `.xml` files are the intermediate representation of your model. You can then `rsync` or `scp` the `.bin` and `.xml` files from wherever you need them!


## Tips for preparing OpenVino

_[this link has helpful information](https://stackoverflow.com/collectives/intel/articles/72141365/how-to-convert-pytorch-model-and-run-it-with-openvino)_

_[also see here](https://github.com/openvinotoolkit/openvino/wiki/CMakeOptionsForCustomCompilation#building-with-custom-opencv)_

After installing opencv and cmake, you can run

```console
git clone https://github.com/openvinotoolkit/openvino.git
git checkout tags/2022.1.0
git submodule update --init --recursive
mkdir build && cd build
```

```console
# only the MYRIAD plugin is supported, and is default "on"
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DTHREADING=SEQ \
  -DOpenCV_DIR="/home/pi/opencv/platforms/linux/arm.toolchain.cmake" \
  -DENABLE_PYTHON=ON \
  -DPYTHON_EXECUTABLE="/usr/bin/python3" \
  -DPYTHON_LIBRARY="/usr/lib/arm-linux-gnueabihf/libpython3.7m.so" \
  -DPYTHON_INCLUDE_DIR="/usr/include/python3.7" \
  -DENABLE_OV_TF_FRONTEND_ENABLE=OFF \
  -DENABLE_OV_PDPD_FRONTEND_ENABLE=OFF \
  -DENABLE_SAMPLES=OFF .. \
  && make --jobs=$(nproc --all)
```

And you have to add `openvino` to the `PYTHONPATH` variable. Add this to `~/.bashrc`:
```console
export PYTHONPATH=/home/pi/openvino/bin/armv7l/Release/lib/python_api/python3.7:$PYTHONPATH
```

## OpenVino Performance Optimizations!

[OpenVino benchmark app](https://docs.openvino.ai/latest/openvino_inference_engine_tools_benchmark_tool_README.html#run-the-tool)


### Measuring Performance

[Getting Performance Numbers](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Getting_Performance_Numbers.html#doxid-openvino-docs-m-o-d-g-getting-performance-numbers)

### Improving Performance

[Training & executing with half-precision?](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/)
- [Intel speaks a bit to this WRT the NCS](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-neural-compute-stick-2-intel-ncs-2-and-16-floating-point-fp16.html)
- [and this also from pytorch](https://pytorch.org/docs/stable/amp.html)
- [specific fp16 optims](https://pytorch.org/docs/stable/notes/amp_examples.html)
- [Great explanation of fp16](https://spell.ml/blog/mixed-precision-training-with-pytorch-Xuk7YBEAACAASJam)

[Preprocess Faster?](https://docs.openvino.ai/latest/openvino_docs_OV_UG_Preprocessing_Details.html#doxid-openvino-docs-o-v-u-g-preprocessing-details)

[General Model Optimization Guide](https://docs.openvino.ai/latest/openvino_docs_model_optimization_guide.html#doxid-openvino-docs-model-optimization-guide)

[Runtime Inference Options](https://docs.openvino.ai/latest/openvino_docs_deployment_optimization_guide_dldt_optimization_guide.html#doxid-openvino-docs-deployment-optimization-guide-dldt-optimization-guide)

[Model Caching or Compiling](https://docs.openvino.ai/latest/openvino_docs_OV_UG_Model_caching_overview.html#doxid-openvino-docs-o-v-u-g-model-caching-overview)
