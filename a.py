import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
X,y = load_boston(return_X_y=True)
#X是特征集，y是预测标签
X_df = pd.DataFrame(X) #转换为dataframe可在Spyder中查看具体数据
X_df = X_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))) 
#这里对每一列特征使用max-min标准化方法，将数值统一到[0,1]区间
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
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
#线性回归的闭式求解
intercept = np.ones((X_train.shape[0], 1)) #设置截距量
X_train1 = pd.DataFrame(np.hstack((intercept, X_train))) #计算X的增广矩阵
m1 = np.dot(X_train1.values.T,X_train1.values) #计算对称矩阵的逆
m2 = np.dot(np.linalg.inv(m1),X_train1.values.T) 
weg_lrc = np.dot(m2,y_train) #得到特征权重
def preout(X_test,weight):
    data_inc = np.hstack((np.ones((X_test.shape[0], 1)),X_test))
    y_pre = np.dot(data_inc,weight)
    return y_pre
#根据训练获得求解得到的权重计算预测值，注意设置截距量这一步
weg_bgd = myBGD(X_train,y_train,num_steps = 50000,learning_rate = 1e-3,add_intercept=True) #设置一个较小的学习率和较多的迭代次数，确保收敛到全局最优附近
weg_sgd = mySGD(X_train,y_train,num_steps = 20000,learning_rate = 1e-3,add_intercept=True)
weg_mgd = myMGD(X_train,y_train,num_steps = 20000,batch_size=5,learning_rate = 1e-3,add_intercept=True)
y_pre0 = preout(X_test,weg_bgd)    
y_pre1 = preout(X_test,weg_sgd)
y_pre2 = preout(X_test,weg_mgd)
#将结果可视化
plt.figure(figsize=(15,10))
plt.plot(y_test,label='True')
plt.plot(y_pre0,label='Pre_myBGD')
plt.plot(y_pre1,label='Pre_mySGD')
plt.plot(y_pre2,label='Pre_myMGD')
#plt.plot(y_pre3,label='Pre_LRC')
plt.legend()
plt.show()
