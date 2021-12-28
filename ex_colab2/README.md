## Stanford課程colab連結
- [colab2](https://colab.research.google.com/drive/1Aa0eKSmyYef1gORvlHv7EeQzSVRb30eL?usp=sharing)

## Files 
- run_colab2.py
    - 實作colab2
- run_test_layers.py
    - 實作nn.Modulelist和nn.Sequential深度學習層的設定

## Note
- How to read and make tensor matrix using SparseTensor?
- [PyTorch 中的 ModuleList 和 Sequential: 区别和使用场景](https://zhuanlan.zhihu.com/p/64990232)

- [nn.moduleList 和Sequential由来、用法和实例 —— 写网络模型](https://blog.csdn.net/e01528/article/details/84397174)

- [When should I use nn.ModuleList and when should I use nn.Sequential?](https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463)
    - Summary: We have to define modulelist for storing the parameter information
    - When defining the optimizer() for that net, you’ll get an error saying that your model has no parameters, because PyTorch does not see the parameters of the layers stored in a Python list. If you use a nn.ModuleList instead, you’ll get no error.
    - Q: Why use ModuleList instead of a normal python list? is it so that parameters are included in the .parameters() iterator?
    - A: Exactly! If you use a plain python list, the parameters won’t be registered properly and you can’t pass them to your optimizer using model.parameters().


## Reference
- [Training Better Deep Learning Models for Structured Data using Semi-supervised Learning](https://towardsdatascience.com/training-better-deep-learning-models-for-structured-data-using-semi-supervised-learning-8acc3b536319)