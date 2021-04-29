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

## 2:考虑到数据集各属性或者说特征之间量纲差异大，对原始据进行标准化：
```python
  X_df = X_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))) 
  #这里对每一列特征使用max-min标准化方法，将数值统一到[0,1]区间
```

## 3:划分训练集和测试集，用来评估算法性能：
```python
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_df, y, random_state=111)
#这里设置了随机种子，确保能够产生一致的训练和测试数据的划分
#也可以自己写函数进行训练集和测试集的划分，示例如下：
def trainTestSplit(X,test_size):
    X_num=X.shape[0]
    train_index=range(X_num)
    test_index=[]
    test_num=int(X_num*test_size)
    for i in range(test_num):
        randomIndex=int(np.random.uniform(0,len(train_index)))
        test_index.append(train_index[randomIndex])
        del train_index[randomIndex]
    train=X.ix[train_index,] 
    test=X.ix[test_index,]
    return train,test
#这里的X是原始数据集或者待划分的数据集，test_size表示测试集所占比例
```
