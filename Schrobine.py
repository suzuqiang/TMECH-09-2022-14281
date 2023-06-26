import os,torch,warnings,math,sys,matplotlib,sklearn
import argparse,datetime
import numpy as np
from MGRN import *
from torch.utils.data import TensorDataset, DataLoader
warnings.filterwarnings('ignore')
#%%Basements
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects


def Parse():
    parser = argparse.ArgumentParser(description='Schrobine')
    parser.add_argument('--PreTrainProNetEpoch', default=100, type=int)
    parser.add_argument('--PreTrainMGRNEpoch', default=50, type=int)
    parser.add_argument('--PLLEpoch', default=100, type=int)
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--data', default='Data1024.npy', type=str,help='数据文件')
    parser.add_argument('--label', default='Label1024.npy', type=str,help='标签文件')
    parser.add_argument('--random', default=False, type=bool,help='是否重新生成随机序列')
    parser.add_argument('--RandomNum', default='RandonNum0.npy', type=str,help='是否重新生成随机序列')
    parser.add_argument('--savemodel', default=True, type=bool,help='是否保存模型')
    parser.add_argument('--shot', default=20, type=int,help='每类数据有标样本数')
    parser.add_argument('--supportusenum', default=8, type=int,help='每类数据有标样本数')
    parser.add_argument('--classnum', default=8, type=int,help='样本类数')
    parser.add_argument('--perclassnum', default=800, type=int,help='每类样本数')
    parser.add_argument('--testnum', default=200, type=int,help='每类测试样本数')
    parser.add_argument('--length', default=30, type=int,help='MGRN采样长度')
    parser.add_argument('--stride', default=3, type=int,help='MGRN采样步长')
    args = parser.parse_args()
    return args
def scatter2D(x, colors):
    palette = np.array(sns.color_palette("hls", np.max(colors)+1))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    txts = []
    # for i in range(8):
    #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)
    return f, ax, sc, txts
def scatter3D(x, colors):
    palette = np.array(sns.color_palette("hls", np.max(colors)+1))
    ax = plt.figure().add_subplot(111, projection = '3d')
    ax.scatter(x[:,0], x[:,1], x[:,2], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    txts = []
    # for i in range(8):
    #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)
    return f, ax, sc, txts
def Getdigits_proj(Feature,n=2):
    feature = []
    for i in range(len(Feature)):
        digits_proj = TSNE(random_state=0,n_jobs=-1,n_components=n).fit_transform(Feature[i])
        feature.append(digits_proj)
    return feature
def Drow(History,Name):
    import matplotlib.pyplot as plt
    x = np.arange(len(History))
    width = 1.5
    fig, ax = plt.subplots(figsize=(8,8),dpi=300)
    for i in range(History.shape[1]):
        ax.bar(x, History[:,i], width,label=Name[i])
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.legend()
    ax.text(.87,-.08,'\nVisualization by DataCharm',transform = ax.transAxes,
            ha='center', va='center',fontsize = 5,color='black',fontweight='bold',family='Roboto Mono')
    plt.show()
class DefineMyDataCenter():
    def __init__(self,Label,args,**kwargs):
        self.args = args
        self.TureLabel = torch.from_numpy(Label).long()
        self.ActuralLabel = torch.from_numpy(Label).long()
        self.ActuralLabel[np.load('Unlabeledtrainlabel.npy')] = -1
        self.LabelUpdate,self.LabelUpdateIndex = np.zeros((len(self.TureLabel),self.args.PreTrainProNetEpoch+\
                                                           self.args.PLLEpoch+10,np.max(Label)+1),dtype='float32')\
            ,np.zeros(len(self.TureLabel),dtype='int32')
        # self.LabelUpdate,self.LabelUpdateIndex = np.load('LabelUpdate4)25)16)38.npy'),100*np.ones(len(self.TureLabel),dtype='int32')
    def GetLabel(self,LabelIndex):
        return self.TureLabel[LabelIndex]
    def SaveData(self):
        np.save('LabelUpdate'+GetTime(),self.LabelUpdate)
    def GetUpdate(self,LabelIndex,New):
        for i in range(len(LabelIndex)):
            self.LabelUpdate[LabelIndex[i]][int(self.LabelUpdateIndex[LabelIndex[i]])] += New[i] 
        self.LabelUpdateIndex[LabelIndex] += 1
    def GetMGRNTrainData(self):
        Querytrainlabel = np.load('Querytrainlabel.npy')
        QueryData = self.LabelUpdate[Querytrainlabel]
        QueryLabel = self.TureLabel[Querytrainlabel].detach().numpy()
        num = (self.args.PreTrainProNetEpoch-self.args.length)//self.args.stride
        Data = np.zeros((len(QueryLabel),num,self.args.length,self.args.classnum),dtype='float32')
        Label = np.zeros((len(QueryLabel),num,self.args.classnum+1),dtype='float32')
        for i in range(len(Label)):
            for n in range(num):
                Data[i,n] = QueryData[i,n*self.args.stride:n*self.args.stride+self.args.length]
                Temp = np.zeros((self.args.classnum+1))
                temp = np.bincount(np.argmax(Data[i,n],1))
                Temp[QueryLabel[i]] = np.equal(QueryLabel[i],np.argmax(temp)) and np.max(temp)>=0.6*self.args.length
                Temp[self.args.classnum] = np.argmax(np.bincount(np.argmax(Data[i,n],1)))
                Label[i,n] = Temp
        Data,Label = Data.reshape(-1,self.args.length,self.args.classnum),Label.reshape(-1,self.args.classnum+1)
        TrainData,TrainLabel=torch.from_numpy(Data).float(),torch.from_numpy(Label).float()
        TrainDataSet = TensorDataset(TrainData,TrainLabel)
        TrainDataSet = DataLoader(dataset=TrainDataSet, batch_size=200, shuffle=True, drop_last=False)
        return TrainDataSet
    def GetPseudoLabel(self,LabelIndex):
        temp = LabelIndex.cpu().detach().numpy()
        labelupdateindex = self.LabelUpdateIndex[LabelIndex][0]
        TempLabelUpdate = self.LabelUpdate[temp,labelupdateindex-self.args.length:labelupdateindex]
        selected,labellndex,pseudolabel = [],[],[]
        for i in range(len(LabelIndex)):
            n = np.max(np.bincount(np.argmax(TempLabelUpdate[i],1)))
            if np.max(np.bincount(np.argmax(TempLabelUpdate[i],1))) > 0.9*self.args.length:
                pseudolabel.append(np.where(np.bincount(np.argmax(TempLabelUpdate[i],1))==n)[0][0])
                selected.append(i)
                labellndex.append(LabelIndex[i])
        return  torch.from_numpy(np.array(selected)).long(),\
                torch.from_numpy(np.array(labellndex)).long(),\
                torch.from_numpy(np.array(pseudolabel)).long()
    
def ProNetGetDataSet(args):
    Data,Label = np.expand_dims(np.float32(np.load(args.data)),1),np.load(args.label)
    MyDataCenter = DefineMyDataCenter(Label,args)
    Label = np.array(range(len(Label)))
    supportraindata,supportrainlabel = np.zeros((args.supportusenum*args.classnum,1,1024),dtype='float32'),np.zeros((args.supportusenum*args.classnum),dtype='int32')
    querytraindata,querytrainlabel = np.zeros(((args.shot-args.supportusenum)*args.classnum,1,1024),dtype='float32'),np.zeros(((args.shot-args.supportusenum)*args.classnum),dtype='int32')
    Unlabeledtraindata,Unlabeledtrainlabel = np.zeros((((args.perclassnum-args.testnum)-args.shot)*args.classnum,1,1024),dtype='float32'),np.zeros((((args.perclassnum-args.testnum)-args.shot)*args.classnum),dtype='int32')
    testdata,testlabel = np.zeros((args.testnum*args.classnum,1,1024),dtype='float32'),np.zeros((args.testnum*args.classnum),dtype='int32')                         
    samplesnum = np.zeros((args.classnum+4),dtype='int32')
    RandomNum = np.load(args.RandomNum)
    for i in range(len(Label)):
        if samplesnum[MyDataCenter.GetLabel(RandomNum[i])] < args.supportusenum:
            supportraindata[samplesnum[8]],supportrainlabel[samplesnum[8]] = Data[RandomNum[i]],Label[RandomNum[i]]
            samplesnum[MyDataCenter.GetLabel(RandomNum[i])],samplesnum[8] = samplesnum[MyDataCenter.GetLabel(RandomNum[i])]+1,samplesnum[8]+1
        elif samplesnum[MyDataCenter.GetLabel(RandomNum[i])] < args.shot:
            querytraindata[samplesnum[9]],querytrainlabel[samplesnum[9]] = Data[RandomNum[i]],Label[RandomNum[i]]
            samplesnum[MyDataCenter.GetLabel(RandomNum[i])],samplesnum[9] = samplesnum[MyDataCenter.GetLabel(RandomNum[i])]+1,samplesnum[9]+1
        elif samplesnum[MyDataCenter.GetLabel(RandomNum[i])] < args.perclassnum-args.testnum:
            Unlabeledtraindata[samplesnum[10]],Unlabeledtrainlabel[samplesnum[10]] = Data[RandomNum[i]],Label[RandomNum[i]]
            samplesnum[MyDataCenter.GetLabel(RandomNum[i])],samplesnum[10] = samplesnum[MyDataCenter.GetLabel(RandomNum[i])]+1,samplesnum[10]+1
        else:
            testdata[samplesnum[11]],testlabel[samplesnum[11]] = Data[RandomNum[i]],Label[RandomNum[i]]
            samplesnum[MyDataCenter.GetLabel(RandomNum[i])],samplesnum[11] = samplesnum[MyDataCenter.GetLabel(RandomNum[i])]+1,samplesnum[11]+1
    del Data, Label, RandomNum, samplesnum
    Supportraindata,Supportrainlabel=torch.from_numpy(supportraindata),torch.from_numpy(supportrainlabel).long()
    SupportTrainDataSet = TensorDataset(Supportraindata,Supportrainlabel)
    SupportTrainDataSet = DataLoader(dataset=SupportTrainDataSet, batch_size=args.supportusenum*args.classnum, shuffle=False, drop_last=False)
    Querytraindata,Querytrainlabel=torch.from_numpy(querytraindata),torch.from_numpy(querytrainlabel).long()
    QueryTrainDataSet = TensorDataset(Querytraindata,Querytrainlabel)
    QueryTrainDataSet = DataLoader(dataset=QueryTrainDataSet, batch_size=args.batch, shuffle=False, drop_last=False)
    Unlabeledtraindata,Unlabeledtrainlabel=torch.from_numpy(Unlabeledtraindata),torch.from_numpy(Unlabeledtrainlabel).long()
    UnlabeledTrainDataSet = TensorDataset(Unlabeledtraindata,Unlabeledtrainlabel)
    UnlabeledTrainDataSet = DataLoader(dataset=UnlabeledTrainDataSet, batch_size=args.batch*5, shuffle=True, drop_last=False)
    Testdata, Testlabel = torch.from_numpy(testdata), torch.from_numpy(testlabel).long()
    TestDataSet = TensorDataset(Testdata, Testlabel)
    TestDataSet = DataLoader(dataset=TestDataSet, batch_size=args.batch, shuffle=True, drop_last=False)
    np.save('Unlabeledtrainlabel',Unlabeledtrainlabel)
    np.save('Querytrainlabel',querytrainlabel)
    return SupportTrainDataSet,QueryTrainDataSet,UnlabeledTrainDataSet,TestDataSet,MyDataCenter

def CNNGetDataSet(args):
    Data,Label = np.expand_dims(np.float32(np.load(args.data)),1),np.load(args.label)
    MyDataCenter = DefineMyDataCenter(Label,args)
    Label = np.array(range(len(Label)))
    traindata,trainlabel = np.zeros((args.shot*args.classnum,1,1024),dtype='float32'),np.zeros((args.shot*args.classnum),dtype='int32')
    unlabeledtraindata,unlabeledtrainlabel = np.zeros((((args.perclassnum-args.testnum)-args.shot)*args.classnum,1,1024),dtype='float32'),np.zeros((((args.perclassnum-args.testnum)-args.shot)*args.classnum),dtype='int32')
    testdata,testlabel = np.zeros((args.testnum*args.classnum,1,1024),dtype='float32'),np.zeros((args.testnum*args.classnum),dtype='int32')                         
    samplesnum = np.zeros((args.classnum+3),dtype='int32')
    RandomNum = np.load(args.RandomNum)
    # RandomNum = np.random.choice(len(Label),len(Label),False)
    # np.save('RandonNum0',RandomNum)
    for i in range(len(Label)):
        if samplesnum[MyDataCenter.GetLabel(RandomNum[i])] < args.shot:
            traindata[samplesnum[8]],trainlabel[samplesnum[8]] = Data[RandomNum[i]],Label[RandomNum[i]]
            samplesnum[MyDataCenter.GetLabel(RandomNum[i])],samplesnum[8] = samplesnum[MyDataCenter.GetLabel(RandomNum[i])]+1,samplesnum[8]+1
        elif samplesnum[MyDataCenter.GetLabel(RandomNum[i])] < args.perclassnum-args.testnum:
            unlabeledtraindata[samplesnum[9]],unlabeledtrainlabel[samplesnum[9]] = Data[RandomNum[i]],Label[RandomNum[i]]
            samplesnum[MyDataCenter.GetLabel(RandomNum[i])],samplesnum[9] = samplesnum[MyDataCenter.GetLabel(RandomNum[i])]+1,samplesnum[9]+1
        else:
            testdata[samplesnum[10]],testlabel[samplesnum[10]] = Data[RandomNum[i]],Label[RandomNum[i]]
            samplesnum[MyDataCenter.GetLabel(RandomNum[i])],samplesnum[10] = samplesnum[MyDataCenter.GetLabel(RandomNum[i])]+1,samplesnum[10]+1
    del Data, Label, RandomNum, samplesnum
    Traindata,Trainlabel=torch.from_numpy(traindata),torch.from_numpy(trainlabel).long()
    TrainDataSet = TensorDataset(Traindata,Trainlabel)
    TrainDataSet = DataLoader(dataset=TrainDataSet, batch_size=args.supportusenum*args.classnum, shuffle=False, drop_last=False)
    Unlabeledtraindata,Unlabeledtrainlabel=torch.from_numpy(unlabeledtraindata),torch.from_numpy(unlabeledtrainlabel).long()
    UnlabeledTrainDataSet = TensorDataset(Unlabeledtraindata,Unlabeledtrainlabel)
    UnlabeledTrainDataSet = DataLoader(dataset=UnlabeledTrainDataSet, batch_size=args.batch, shuffle=True, drop_last=False)
    Testdata,Testlabel = torch.from_numpy(testdata),torch.from_numpy(testlabel).long()
    TestDataSet = TensorDataset(Testdata, Testlabel)
    TestDataSet = DataLoader(dataset=TestDataSet, batch_size=args.batch, shuffle=True, drop_last=False)
    return TrainDataSet,UnlabeledTrainDataSet,TestDataSet,MyDataCenter

def SiaNetGetDataSet(args):
    Data,Label = np.expand_dims(np.float32(np.load(args.data)),1),np.load(args.label)
    MyDataCenter = DefineMyDataCenter(Label,args)
    Label = np.array(range(len(Label)))
    supportraindata,supportrainlabel = np.zeros((args.supportusenum*args.classnum,1,1024),dtype='float32'),np.zeros((args.supportusenum*args.classnum),dtype='int32')
    querytraindata,querytrainlabel = np.zeros(((args.shot-args.supportusenum)*args.classnum,1,1024),dtype='float32'),np.zeros(((args.shot-args.supportusenum)*args.classnum),dtype='int32')
    Unlabeledtraindata,Unlabeledtrainlabel = np.zeros((((args.perclassnum-args.testnum)-args.shot)*args.classnum,1,1024),dtype='float32'),np.zeros((((args.perclassnum-args.testnum)-args.shot)*args.classnum),dtype='int32')
    testdata,testlabel = np.zeros((args.testnum*args.classnum,1,1024),dtype='float32'),np.zeros((args.testnum*args.classnum),dtype='int32')                         
    samplesnum = np.zeros((args.classnum+4),dtype='int32')
    RandomNum = np.load(args.RandomNum)
    for i in range(len(Label)):
        if samplesnum[MyDataCenter.GetLabel(RandomNum[i])] < 1:
            supportraindata[samplesnum[8]],supportrainlabel[samplesnum[8]] = Data[RandomNum[i]],Label[RandomNum[i]]
            samplesnum[MyDataCenter.GetLabel(RandomNum[i])],samplesnum[8] = samplesnum[MyDataCenter.GetLabel(RandomNum[i])]+1,samplesnum[8]+1
        elif samplesnum[MyDataCenter.GetLabel(RandomNum[i])] < args.shot:
            querytraindata[samplesnum[9]],querytrainlabel[samplesnum[9]] = Data[RandomNum[i]],Label[RandomNum[i]]
            samplesnum[MyDataCenter.GetLabel(RandomNum[i])],samplesnum[9] = samplesnum[MyDataCenter.GetLabel(RandomNum[i])]+1,samplesnum[9]+1
        elif samplesnum[MyDataCenter.GetLabel(RandomNum[i])] < args.perclassnum-args.testnum:
            Unlabeledtraindata[samplesnum[10]],Unlabeledtrainlabel[samplesnum[10]] = Data[RandomNum[i]],Label[RandomNum[i]]
            samplesnum[MyDataCenter.GetLabel(RandomNum[i])],samplesnum[10] = samplesnum[MyDataCenter.GetLabel(RandomNum[i])]+1,samplesnum[10]+1
        else:
            testdata[samplesnum[11]],testlabel[samplesnum[11]] = Data[RandomNum[i]],Label[RandomNum[i]]
            samplesnum[MyDataCenter.GetLabel(RandomNum[i])],samplesnum[11] = samplesnum[MyDataCenter.GetLabel(RandomNum[i])]+1,samplesnum[11]+1
    del Data, Label, RandomNum, samplesnum
    Supportraindata,Supportrainlabel=torch.from_numpy(supportraindata),torch.from_numpy(supportrainlabel).long()
    SupportTrainDataSet = TensorDataset(Supportraindata,Supportrainlabel)
    SupportTrainDataSet = DataLoader(dataset=SupportTrainDataSet, batch_size=args.supportusenum*args.classnum, shuffle=False, drop_last=False)
    Querytraindata,Querytrainlabel=torch.from_numpy(querytraindata),torch.from_numpy(querytrainlabel).long()
    QueryTrainDataSet = TensorDataset(Querytraindata,Querytrainlabel)
    QueryTrainDataSet = DataLoader(dataset=QueryTrainDataSet, batch_size=args.batch, shuffle=False, drop_last=False)
    Unlabeledtraindata,Unlabeledtrainlabel=torch.from_numpy(Unlabeledtraindata),torch.from_numpy(Unlabeledtrainlabel).long()
    UnlabeledTrainDataSet = TensorDataset(Unlabeledtraindata,Unlabeledtrainlabel)
    UnlabeledTrainDataSet = DataLoader(dataset=UnlabeledTrainDataSet, batch_size=args.batch, shuffle=True, drop_last=False)
    Testdata, Testlabel = torch.from_numpy(testdata), torch.from_numpy(testlabel).long()
    TestDataSet = TensorDataset(Testdata, Testlabel)
    TestDataSet = DataLoader(dataset=TestDataSet, batch_size=args.batch, shuffle=True, drop_last=False)
    return SupportTrainDataSet,QueryTrainDataSet,UnlabeledTrainDataSet,TestDataSet,MyDataCenter
      
def GetSomeLayer(LayerName,in_channels,out_channels=0,Limit=False):
    """
        GetSomeLayer(LayerName,in_channels,out_channels):
            :param LayerName: 网络层名
            :param in_channels,out_channels: 输入输出通道数
            :return:返回网络层
    """
    if LayerName == 'Blocks1DTorch':
        Layer = Blocks1DTorch(in_channels,out_channels)
    elif LayerName == 'LinearLayer':
        Layer = LinearLayer(in_channels,out_channels)
    elif LayerName == 'ClasserTorch':
        Layer = ClasserTorch(in_channels,out_channels)
    LayerName = 'InitialModule/'+LayerName+'_'+str(in_channels)+'_'+str(out_channels)+'.pkl'
    if os.path.exists(LayerName):
        Layer.load_state_dict(torch.load(LayerName))
    else:
        torch.save(Layer.state_dict(),LayerName)
    return Layer
def GetTime():
    """
        GetTime():
            :return:时间戳[月)日)时)分]
    """
    time = datetime.datetime.now()
    month = time.month
    day = time.day
    hour = time.hour
    minute = time.minute
    now = str(month)+')'+str(day)+')'+str(hour)+')'+str(minute)
    return now
def SetSeed():
    seed_value = 1
    np.random.seed(seed_value)
    torch.manual_seed(seed_value) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子 
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def GetLoss(logits,label,device):
    """
        GetLoss(logits,label):
            :param logits: 预测向量
            :param label: 标签
            :return: 损失和
    """
    Loss = torch.zeros((1))
    Loss = Loss.to(device)
    for i in range(len(label)):
        temp = torch.log10(logits[i,label[i].cpu().detach().numpy()])
        if not(torch.isnan(temp) or torch.isinf(temp)):
            Loss -= temp
    return Loss
def GetAccuracy(logits,label):
    """
        GetLoss(logits,label):
            :param logits: 预测向量
            :param label: 标签
            :return: 正确预测数
    """
    return np.sum(np.array(np.equal(label.cpu().detach().numpy(),logits.max(1).indices.cpu().detach().numpy())))
class Blocks1DTorch(torch.nn.Module):
    """
        __init__(self, in_channels, out_channels):
            :in_channels, out_channels: 输入输出通道数
        forward(self, inputs):
            :param inputs: 输入特征 (参考形状:[batch,channel,length])
            :return:输出特征
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Blocks1DTorch, self).__init__(**kwargs)
        self.conv2d11 = torch.nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0)
        self.conv2d13 = torch.nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
        self.conv2d15 = torch.nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=5,stride=1,padding=2)
        self.conv2d17 = torch.nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=7,stride=1,padding=3)
        self.Blocks = torch.nn.Sequential(torch.nn.MaxPool1d(kernel_size=2,stride=2,padding=0),torch.nn.BatchNorm1d(num_features=out_channels),torch.nn.ReLU())
    def forward(self, inputs, training=None):
        return self.Blocks(self.conv2d11(inputs)+self.conv2d13(inputs)+self.conv2d15(inputs)+self.conv2d17(inputs))
class LinearLayer(torch.nn.Module):
    """
        __init__(self, in_channels, out_channels):
            :in_channels, out_channels: 输入特征点数, 类别数
        forward(self, inputs):
            :param inputs: 输入特征 (参考形状:[batch,channel,height,width])
            :return: prediction
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(LinearLayer, self).__init__( **kwargs)
        self.Blocks = torch.nn.Sequential(
            torch.nn.Linear(in_features = in_channels, out_features = out_channels),
            torch.nn.BatchNorm1d(num_features=out_channels),torch.nn.ReLU(),torch.nn.Dropout(0.5))
    def forward(self, inputs, training=None):
        out = self.Blocks(torch.flatten(inputs,1))
        return out
class ClasserTorch(torch.nn.Module):
    """
        __init__(self, in_channels, out_channels):
            :in_channels, out_channels: 输入特征点数, 类别数
        forward(self, inputs):
            :param inputs: 输入特征 (参考形状:[batch,channel,height,width])
            :return: prediction
    """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ClasserTorch, self).__init__( **kwargs)
        self.Blocks = torch.nn.Sequential(
            torch.nn.Linear(in_features = in_channels, out_features = 1024),
            torch.nn.BatchNorm1d(num_features=1024),torch.nn.ReLU(),
            torch.nn.Linear(in_features = 1024, out_features = 128),
            torch.nn.BatchNorm1d(num_features=128),torch.nn.ReLU(),
            torch.nn.Linear(in_features = 128, out_features = out_channels),torch.nn.Softmax(1))
    def forward(self, inputs, training=None):
        out = self.Blocks(torch.flatten(inputs,1))
        return out
#%%样例
#运行样例测试模块时应将相应自输入模块设置为cpu,调用模块进行训练时设置为cuda
if __name__ == '__main__':
    import tensorflow as tf
    args = Parse()
    batch,channels,outchannels,hight,width = 7,37,128,128,256
    Data1D = np.float32(np.random.uniform(-50,50,(batch,channels,hight)))
    Torch1D = torch.from_numpy(Data1D)
    Data1D = Data1D.transpose(0,2,1)
    Tensorflow1D = tf.convert_to_tensor(Data1D)
    Data2D = np.float32(np.random.uniform(-50,50,(batch,channels,hight,width)))
    Torch2D = torch.from_numpy(Data2D)
    Data2D = Data2D.transpose(0,2,3,1)
    Tensorflow2D = tf.convert_to_tensor(Data2D)
    del Data1D
    del Data2D