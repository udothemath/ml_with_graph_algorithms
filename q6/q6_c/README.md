## Task
實作GraphSage並測試計算加速前後的效能比較

## Installation @AI cloud for GPU computation
- pip install torch==1.9.0 -f https://download.pytorch.org/whl/torch_stable.html

## Result
![perf_v0_cpu](https://user-images.githubusercontent.com/10674490/147724036-6a292b6d-9639-4289-8e4f-33594c02011b.png)
![perf_v1_gpu](https://user-images.githubusercontent.com/10674490/147724040-38de823f-5cbd-4419-9818-e10d8e3cd08a.png)

## ToDo
- Optimize the setting of number of worker
- Utilize pytorch lightning
- Implement different framework

## Ref
- [post: Sampling Large Graphs in PyTorch Geometric](https://towardsdatascience.com/sampling-large-graphs-in-pytorch-geometric-97a6119c41f9)
    - setting for number of worker
- [package: PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
- [paper with code: GraphSAGE](https://paperswithcode.com/method/graphsage)
- [paper: ACCELERATING TRAINING AND INFERENCE OF GRAPH NEURAL NETWORKS WITH FAST SAMPLING AND PIPELINING](https://arxiv.org/pdf/2110.08450.pdf)
    - Setting for multi-GPU
- [A Comprehensive Case-Study of GraphSage with Hands-on-Experience using PyTorchGeometric Library and Open-Graph-Benchmark’s Amazon Product Recommendation Dataset](https://towardsdatascience.com/a-comprehensive-case-study-of-graphsage-algorithm-with-hands-on-experience-using-pytorchgeometric-6fc631ab1067)
    - [colab](https://colab.research.google.com/github/sachinsharma9780/interactive_tutorials/blob/master/notebooks/example_output/Comprehensive_GraphSage_Guide_with_PyTorchGeometric_Output.ipynb#scrollTo=PTvt6kQYnhXz)

## Question
- [W ParallelNative.cpp:214] Warning: Cannot set number of intraop threads after parallel work has started or after set_num_threads call when using native parallel backend (function set_num_threads)
    - Summary: If you can access your dataloaders, set num_workers=0 when creating a dataloader. We cannot use multiple workers while loading the dataset
    - [Pytorch : W ParallelNative.cpp:206](https://stackoverflow.com/questions/64772335/pytorch-w-parallelnative-cpp206)
