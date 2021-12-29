## Task
實作GraphSage並測試計算加速前後的效能比較

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