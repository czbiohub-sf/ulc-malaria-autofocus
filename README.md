# Single-Shot Auto Focus (SSAF) for the ULC Malaria Scope

## Installation

These instructions are for installation onto the BRUNO HPC at the Chan-Zuckerberg Biohub.

Create (or activate) a conda environment if you have not already.

Install PyTorch and Torchvision from [here](https://pytorch.org/get-started/locally/), selecting the installation settings for

- Stable
- Linux
- Conda
- Python
- Cuda (most recent, or the Cuda version that the HPC uses)

and the command should look something like

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

I most likely missed a package or two, as I am writing this post-hoc. If you find that you need a package that I have  not included here, please create an issue!


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

which should create a `.onnx` file. Then run

```console
mo --input_model <PATH_TO_YOUR_onnx>
```

You should then have a `.bin` and a `.xml` file, which act as the intermediate representation of your model. You can then `rsync` or `scp` the `.bin` and `.xml` files from wherever you need them!



## Low Signal-to-noise ratio preparation of OpenVino

__vaguely following https://github.com/openvinotoolkit/openvino/wiki/CMakeOptionsForCustomCompilation#building-with-custom-opencv__

After installing opencv and cmake, you can run

```console
git clone https://github.com/openvinotoolkit/openvino.git
git checkout tags/2022.1.0
git submodule update --init --recursive
mkdir build && cd build
```

```
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

And we have to add `openvino` to the `PYTHONPATH` variable. Add this to `~/.bashrc`:
```console
export PYTHONPATH=/home/pi/openvino/bin/armv7l/Release/lib/python_api/python3.7:$PYTHONPATH
```


## OpenVino Performance Optimizations!

### Measuring Performance

[Getting Performance Numbers](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Getting_Performance_Numbers.html#doxid-openvino-docs-m-o-d-g-getting-performance-numbers)

### Improving Performance

[Training & executing with half-precision?](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/)
- [Intel speaks a bit to this WRT the NCS](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-neural-compute-stick-2-intel-ncs-2-and-16-floating-point-fp16.html)
- [and this also from pytorch](https://pytorch.org/docs/stable/amp.html)

[Preprocess Faster?](https://docs.openvino.ai/latest/openvino_docs_OV_UG_Preprocessing_Details.html#doxid-openvino-docs-o-v-u-g-preprocessing-details)

[General Model Optimization Guide](https://docs.openvino.ai/latest/openvino_docs_model_optimization_guide.html#doxid-openvino-docs-model-optimization-guide)

[Runtime Inference Options](https://docs.openvino.ai/latest/openvino_docs_deployment_optimization_guide_dldt_optimization_guide.html#doxid-openvino-docs-deployment-optimization-guide-dldt-optimization-guide)

[Model Caching or Compiling](https://docs.openvino.ai/latest/openvino_docs_OV_UG_Model_caching_overview.html#doxid-openvino-docs-o-v-u-g-model-caching-overview)
