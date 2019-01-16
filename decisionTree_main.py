from tkinter import *
import tkinter.messagebox
from tkinter import ttk
from tkinter.filedialog import askdirectory
import numpy as np
import math
import csv
import operator
from graphviz import Digraph
from PIL import Image
from PIL import ImageTk

Trainfilename = '' #训练集文件名
Testfilename = '' #验证集文件名
decisionTree = {} #决策树
comboxList = {} #下拉框列表
prefeatures = list() #属性列
predataSet = list() #数据集
A = Digraph() #图
cnt = 0

def createDataSet(): #创建数据集.
    getfile = open(Trainfilename,'r',encoding='utf-8-sig') #读入数据文件
    dataSet = list(csv.reader(getfile)) #将csv文件读入转换成List
    features = dataSet[0] #获取第一行的标签
    features.pop(-1) #属性不包括最后一项.
    featurelist = {} #存储每一个属性对应的表现形式集合.
    del dataSet[0] #数据集不包括第一行
    for i in range(len(features)): #读取每一个属性的表现形式.
        tmpset = set()
        for cur in dataSet:
            tmpset.add(cur[i])
        featurelist[features[i]] = tmpset
    getfile.close()
    return dataSet,features,featurelist

def calEnt(dataSet): #计算一个数据集的信息熵
    rowsum = len(dataSet) #数据集总行数
    labelCount = {} #记录每一个标签对应出现的次数.
    for curdata in dataSet: #提取每一行
        curlabel = curdata[-1] #提取最后一个元素.
        if curlabel not in labelCount.keys(): #判断是否在字典里面出现过,没有出现就赋值为0.
            labelCount[curlabel] = 0
        labelCount[curlabel] += 1 #更新出现次数.
    Ent = 0.0
    for id in labelCount: #计算信息熵
        prob = float(labelCount[id]) / rowsum #计算选择这个标签的概率.
        Ent -= prob * math.log(prob,2)
    return Ent

def splitDataSet(dataSet,id,val): #对于给定的数据集进行特定的划分.
    curdataSet = [] 
    for curVec in dataSet:
        if curVec[id] == val :
            finVec = curVec[:id] #返回的数据集不需要这个属性了.所以要去掉.
            finVec.extend(curVec[id+1:])
            curdataSet.append(finVec)
    return curdataSet

def getBestFeature(dataSet): #获取最优划分属性
    featurenum = len(dataSet[0]) - 1 #得到特征的数量
    gain_pre=calEnt(dataSet) #计算一次总的信息熵
    Best_gain = 0.0 #记录最大的信息增益
    Best_index = -1 #记录最优特征的下标
    for id in range(featurenum): #遍历所有的特征.
        uniquefeat = set()
        for perf in dataSet:
            uniquefeat.add(perf[id]) #找出第id个特征的所有可能表现形式存储到set里面
        gain_now = gain_pre #记录当前的信息增益
        for val in uniquefeat:
            subdataSet = splitDataSet(dataSet,id,val) #进行根据每个表现形式进行划分
            prob = float(len(subdataSet)) / len(dataSet) #计算子集的概率
            gain_now -= prob * calEnt(subdataSet) #计算信息增益.
        #print("The %dth information gain: %.3f" % (id,gain_now))
        if(gain_now > Best_gain): #更新最大增益.
            Best_gain = gain_now
            Best_index = id
    return Best_index

def getMajor(decisionlist): #对于给定的结果集返回出现次数最多的结果
    dict_list = {}
    for cur in decisionlist: #统计一下每个结果出现的次数
        if cur not in dict_list.keys():
            dict_list[cur] = 0
        dict_list[cur] += 1
    nowlist = sorted(dict_list.items(), key = operator.itemgetter(1), reverse = True) #按照字典的值降序排列
    return nowlist[0][0] #返回出现次数最多的结果

def DecisionTree(dataSet,labels,featurelist): #生成决策树的子树
    decision_of_res = [] #记录当前数据集里面关于的结果集情况.
    for cur in dataSet: #统计结果集.
        decision_of_res.append(cur[-1])
    if decision_of_res.count(decision_of_res[0]) == len(decision_of_res): #数据集属于同一类别即无需划分了.
        return decision_of_res[0] #此时直接返回对应的结果即可.
    if len(dataSet[0]) == 1: #属性集为空,就不需要划分了.
        return getMajor(decision_of_res) #此时返回出现次数最多的结果.
    Bestfeat = getBestFeature(dataSet) #获取最优属性特征的下标
    Bestfeatlabel = labels[Bestfeat] #获取最优属性特征的标签
    decisiontree = {Bestfeatlabel:{}} #初始化当前这棵树
    tmplabels = list(labels) 
    del(tmplabels[Bestfeat]) #删除已经使用的标签
    for val in featurelist[Bestfeatlabel]: #遍历每一个结果,分别创建决策树
        nextdataSet = splitDataSet(dataSet,Bestfeat,val)
        if len(nextdataSet) == 0: #如果数据集没有这个属性
            decisiontree[Bestfeatlabel][val] = getMajor(decision_of_res)
        else :
            decisiontree[Bestfeatlabel][val] = DecisionTree(nextdataSet,tmplabels,featurelist)
    return decisiontree

def DFS(Tree,father,text): #DFS决策树画图.
    global cnt
    if isinstance(Tree,dict): #如果不是叶节点,那么就遍历这个字典生成其他子树.
        if father == '-1': #为父亲节点的时候,添加点,不添加边.
            for key,value in Tree.items():
                kkey = str(cnt) 
                cnt += 1
                A.node(kkey,key,fontname='Simhei') #节点的标识符和节点的内容.
                DFS(value,kkey,'')
        elif text == '' :  #text为空表示当前不需要添加点和边,直接继续DFS
            for key,value in Tree.items():
                DFS(value,father,key)
        else : #text不为' '表示当前需要添加点和边.
            for key,value in Tree.items():
                kkey = str(cnt)
                cnt += 1
                A.node(kkey,key,fontname='Simhei')
                A.edge(father,kkey,text,color='cyan',fontname='Simhei')
                DFS(value,kkey,'')
    else : #是叶节点,直接返回对应结果.
        sstr = Tree+str(cnt)
        cnt += 1
        A.node(sstr,str(Tree),color='green',fontname='Simhei')
        if father !='-1':
            A.edge(father,sstr,text,color='cyan',fontname='Simhei')

def getIndex(key): #获取属性对应在数据集里面的下标
    global prefeatures
    for i in range(len(prefeatures)):
        if prefeatures[i] == key:
            return i

def getres(Tree,data): #获取一个数据在决策树上决策后的结果
    if isinstance(Tree,dict): #如果非叶子节点就遍历找出合法的第一个
        for key,value in Tree.items():
            index = getIndex(key)
            return getres(value[data[index]],data)
    else : #叶子节点直接返回结果.
        return Tree 

def getname_of_file(): #获取数据集的名字.
    global Trainfilename
    Trainfilename = tkinter.filedialog.askopenfilename()
    if Trainfilename != '':
        tkinter.messagebox.showinfo('提醒','导入成功!')

def getTree(): #生成决策树.
    if Trainfilename == '':
        tkinter.messagebox.showerror('错误','未导入训练数据!')
    else :
        global prefeatures,decisionTree,predataSet
        predataSet,prefeatures,featurelist = createDataSet() #读取数据
        decisionTree = DecisionTree(predataSet,prefeatures,featurelist) #生成决策树
        label = Label(root,text='判定给定西瓜是否是好瓜的决策树.',width=30,height=40,fg='red',font=('华文新魏',20))
        label.place(relx=0.35, rely=0.03, anchor=CENTER)
        DFS(decisionTree,'-1','') #遍历决策树
        A = np.load('graph.npy').item()
        A.render(filename='img',cleanup=True,format='png') #生成图片存储为gif
        img = Image.open('img.png')
        photoImg = ImageTk.PhotoImage(img)
        canvas = Canvas(root,width = img.size[0]+10,height = img.size[1]+10,bg='white') #创建画布
        canvas.place(relx=0.010, rely=0.06)#,relwidth=0.70, relheight=0.85)
        canvas.create_image(248,260,image = photoImg) #在画布里面加载图片
        root.mainloop()

def calrate_of_test(): #返回决策树正确率
    getfile = open(Testfilename,'r',encoding='utf-8-sig') #读入数据文件
    dataSet = list(csv.reader(getfile)) #将csv文件读入转换成List
    count_for_true = 0
    for val in dataSet:
        if  getres(decisionTree,val) == val[-1]:
            count_for_true += 1
    return len(dataSet),count_for_true,1.0*count_for_true/(len(dataSet))*100

def getverify(): #进行决策树正确率的计算
    global Testfilename
    Testfilename = tkinter.filedialog.askopenfilename()
    if Testfilename == '':
        return 
    res_all,res_true,res_rate = calrate_of_test()
    windows=Tk()
    windows.title("验证集正确率")
    canvas=Canvas(windows,height=500,width=500) # 设置画布样式
    canvas.pack() # 将画布打包到窗口
    #利用画布的create_arc画饼形，(400,400)和(100,100)为饼形外围的矩形,
    # start=角度起始，extent=旋转的度数，fill=填充的颜色
    tp = 360.0/res_all*res_true #计算占的份数
    canvas.create_arc(400,400,100,100,start=0,extent=360-tp,fill="red")
    canvas.create_arc(400,400,100,100,start=360-tp,extent=tp,fill="green")
    # 为各个扇形添加内容，圆心为(250，250)
    canvas.create_text(410,220,text='错误率'+str(round(100.0-round(res_rate,2),2))+'%',font=('华文新魏',20))
    canvas.create_text(330,100,text='正确率'+str(round(res_rate,2))+'%',font=('华文新魏',20))
    # 开启消息循环
    windows.mainloop()

def getcombox(win,data,id): #根据给定的data和位置创建一个下拉框
    comvalue=tkinter.StringVar()#窗体自带的文本，新建一个值
    comboxlist=ttk.Combobox(win,textvariable=comvalue,value=list(data)) #初始化
    comboxlist.current(1)  #选择第一个
    comboxlist.place(relx=0.49, rely=0.19+0.10*float(id), anchor=CENTER)
    return comboxlist

def getLabel(win,labs,id): #根据给定的labs和位置创建一个标签
    label = Label(win,width=7,height=2,text=labs+':',fg='red')
    label.place(relx=0.09, rely=0.19+0.10*float(id), anchor=CENTER)

def getres_of_test(): #返回下拉框里面的数据对应的决策结果
    data_of_list = []
    global comboxList
    for i in range(len(prefeatures)):
        data_of_list.append(comboxList[i].get())
    tkinter.messagebox.showinfo('结果',getres(decisionTree,data_of_list))

def gettest(): #输入数据进行决策.
    win = Tk()
    #win.wm_attributes('-topmost',1)
    win.title("决策树数据输入")    # 添加标题
    win.geometry('310x310')
    comboxset = {}
    global comboxList
    for i in range(len(prefeatures)):
        comboxset[i] = set()
        for inx in range(len(predataSet)):
            comboxset[i].add(predataSet[inx][i])
        getLabel(win,prefeatures[i],i)
        comboxList[i] = getcombox(win,comboxset[i],i)
    Btn_of_ttest = Button(win, text ='生成决策结果', command = getres_of_test,fg='green',relief='groove')
    Btn_of_ttest.place(relx=0.34, rely=0.19+0.10*(len(prefeatures)), relwidth=0.28, relheight=0.13)
    win.mainloop()

if __name__ == "__main__":
    root = Tk() # 主窗口
    root.title('决策树')
    root.geometry('710x610')
    root.resizable(False,False)
    Btn_of_getname = Button(root, text ='导入训练集', command = getname_of_file,fg='green',relief='solid')
    Btn_of_getname.place(relx=0.75, rely=0.07, relwidth=0.2, relheight=0.1)
    Btn_of_getTree = Button(root, text ='生成决策树', command = getTree,fg='green',relief='solid')
    Btn_of_getTree.place(relx=0.75, rely=0.31, relwidth=0.2, relheight=0.1)
    Btn_of_verify = Button(root, text ='计算决策树正确率', command = getverify,fg='green',relief='solid')
    Btn_of_verify.place(relx=0.75, rely=0.56, relwidth=0.2, relheight=0.1)
    Btn_of_test = Button(root, text ='测试数据', command = gettest,fg='green',relief='solid')
    Btn_of_test.place(relx=0.75, rely=0.80, relwidth=0.2, relheight=0.1)
    root.mainloop()