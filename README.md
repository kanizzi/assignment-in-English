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
## 4:建立线性回归模型。首先明确线性回归就是在最小化均方误差的约束下，求解不同特征对于预测目标变量的贡献或者说权重。我们通过不同的梯度下降方法来实现权重参数的求解：
```python
#定义批量梯度下降函数求取特征权重
def myBGD(features,target,num_steps,learning_rate,add_intercept=False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features)) 
        #这里是设置截距量，类比一次线性回归函数中y=ax+b中的b
    weights = np.zeros(features.shape[1])  #初始化特征权重
    loss_list = []
    epochs_list = []
    for step in range(num_steps):
        pre = np.dot(features, weights)
        loss = pre - target
        loss_list.append(np.mean(np.abs(loss)))
        gradient = np.dot(features.T, loss) / features.shape[0]
        weights  = weights - learning_rate * gradient
        epochs_list.append(step)
        if step % 5000 == 0:
                print ('Regression Loss is:',np.mean(np.abs(loss)))
    #plt.figure(figsize=(16,9))
    plt.plot(epochs_list,loss_list)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Batch Gradient Descent Loss Curve')
    plt.savefig('loss_bgd.tiff',dpi=300,format='tif')
    plt.show()
    return weights
    
###定义随机梯度下降函数求取特征权重
def mySGD(features,target,num_steps,learning_rate,add_intercept=False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = pd.DataFrame(np.hstack((intercept, features)))
        
    weights = np.zeros(features.shape[1])  #初始化特征权重
    loss_list = []
    epochs_list = []
    for step in range(num_steps):
        pre = np.dot(features, weights)
        loss = pre - target   #如果是target-pre,则后面更新权重应为：weights + learning_rate * gradient
        loss_list.append(np.mean(np.abs(loss)))
        inx = np.random.randint(features.shape[0])
        #features = features.reset_index(drop=True)
        gradient = np.dot(features.T[inx], loss[inx])
        weights  = weights - learning_rate * gradient
        epochs_list.append(step)
        if step % 2000 == 0:
                print ('Regression Loss is:',np.mean(np.abs(loss)))
    plt.plot(epochs_list,loss_list)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Stochastic Gradient Descent Loss Curve')
    plt.savefig('loss_sgd.tiff',dpi=300,format='tif')
    plt.show()
    return weights

###定义mini-batch梯度下降函数求取特征权重
def myMGD(features,target,num_steps,batch_size,learning_rate,add_intercept=False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = pd.DataFrame(np.hstack((intercept, features)))
        
    weights = np.zeros(features.shape[1])  #初始化特征权重
    loss_list = []
    epochs_list = []
    
    for step in range(num_steps):
        pre = np.dot(features, weights)
        loss = pre - target   
        loss_list.append(np.mean(np.abs(loss)))
        inx = np.random.choice(range(features.shape[0]), batch_size, replace=False)
        #features = features.reset_index(drop=True)
        gradient = np.dot(features.T[inx], loss[inx])
        weights  = weights - learning_rate * gradient
        epochs_list.append(step)
        if step % 2000 == 0:
                print ('Regression Loss is:',np.mean(np.abs(loss)))
    plt.plot(epochs_list,loss_list)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Mini-batch Gradient Descent Loss Curve')
    plt.savefig('loss_mgd.tiff',dpi=300,format='tif')
    plt.show()
    return weights
```
<font  size=2 >*线性回归是机器学习中最基本的问题之一，通过全面了解线性回归的本质以及实现，我们可以收获关于机器学习中的一些基本思路和方法。*</font>

## 5:对于线性回归问题，除了梯度下降算法能够去不断拟合特征权重，我们还可以通过矩阵的方式求解原问题的闭式结果或者说解析解。直接给出推导结果：
(https://github.com/kanizzi/assignment-in-English/blob/main/20200928110441987.png)
