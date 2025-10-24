# For blog 2 reproduction

Built on GH200

Cutlass: '/home/nvidia/sglang/nvidia_cutlass_dsl-4.2.1.dev1-cp312-cp312-manylinux_2_28_aarch64.whl'

sgl-kernel: '/home/nvidia/sglang/sgl-kernel/dist/sgl_kernel-0.3.10-cp310-abi3-manylinux2014_aarch64.whl'

docker build   --build-arg CUDA_VERSION=12.9.1   --build-arg BUILD_TYPE=all   --build-arg BRANCH_TYPE=local   --build-arg DEEPEP_COMMIT=1fd57b0276311d035d16176bb0076426166e52f3   --build-arg FLASHINFER_COMMIT=68b8e6db730e81c9ac5ae3ed55b4b01dde5df0a5   --build-arg CMAKE_BUILD_PARALLEL_LEVEL=2   -t sglang-custom:0.0.2   -f docker/Dockerfile .
           