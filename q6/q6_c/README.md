## Last updated: 2022/1/10

## Task
實作GraphSage並測試計算加速前後的效能比較

## Execution @AI cloud for GPU computation
若(第一次)要在行內研發環境執行，請依序進行以下三步驟
1. 環境設定　　
2. 虛擬環境設定
3. 安裝套件　　　　
 
設定完成後，下次直接啟用虛擬環境即可套用設定。　　

### 環境設定
- 選取instance type with GPU       
    <img src="https://user-images.githubusercontent.com/10674490/147896458-953cd4e2-21cb-4c5f-ba1a-c1c869ffaf0a.png" height="200">

- 選取image   
    - 建議使用Option GPU Cuda10        
    <img src="https://user-images.githubusercontent.com/10674490/147896477-2ddb77ae-37d2-44dd-8a85-057fa6fa07d5.png" height="200">    
### 虛擬環境設定
- 安裝virtual environment    
    $ python3 -m venv <name_of_your_virtual_env>    
    舉例，如果你要把虛擬環境安裝在路徑/home/jovyan，虛擬環境的名稱為torch_190，則使用    
    $ python3 -m venv /home/jovyan/torch_190
- 啟動虛擬環境(activate)       
    $ source /home/jovyan/torch_190/bin/activate
- 不需要使用時，退出虛擬環境(deactivate)    
    $ deactivate    
- 參考資料：[虛擬環境與套件](https://docs.python.org/zh-tw/3/tutorial/venv.html)    

### 安裝套件
- bash requirements_torch.sh

## Result
- cpu    
    ![perf_v0_cpu](https://user-images.githubusercontent.com/10674490/147724036-6a292b6d-9639-4289-8e4f-33594c02011b.png)
- gpu    
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
