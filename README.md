# Python 机器学习_线性回归
[参考资料](https://blog.csdn.net/kepengs/article/details/84812666?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_baidulandingword-0&spm=1001.2101.3001.4242)

<font face="黑体" color=green size=5>我是黑体，绿色，尺寸为5</font>

<font size=1>字体大小size=1</font>
<font size=3>字体大小size=3</font>
<font size=5>字体大小size=5</font>

<font  size=2 >**线性回归是机器学习中最基本的问题之一，通过全面了解线性回归的本质以及实现，我们可以收获关于机器学习中的一些基本思路和方法。**</font>

## 1:首先我们以sklearn自带的波士顿房价数据集为对象， 载入数据：
```python
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.datasets import load_boston
  X,y = load_boston(return_X_y=True)
  #X是特征集，y是预测标签
  X_df = pd.DataFrame(X) #转换为dataframe可在Spyder中查看具体数据
```
