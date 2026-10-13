Model Weights and Deployment
Raw model weight binaries are excluded from this source repository.
- Training Source: Trained via Huawei Cloud ModelArts using synchronized RGB and thermal disaster datasets.
- Quantization & Compilation: Optimized via MindSpore and converted to low-precision formats for Huawei Ascend 310 execution.
- Deployment: Distributed dynamically to edge nodes via Huawei Intelligent EdgeFabric (IEF) during runtime initialization.