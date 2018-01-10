#coding=utf8
import json
import csv
import talib
import numpy as np
f=open("stockData.csv")
Data=[]
stkcode='SH600000'
temp=[]
reader = csv.reader(f)
for line in reader:
    if stkcode == line[0]:
        temp.append(line)
    else:
        Data.append(temp)
        temp=[]
        stkcode=line[0]
f.close()


mode=1 #用开盘、收盘的平均值表示K线位置
min_len=15 #最小时间跨度
max_len=60 #最大时间跨度
######## 以下所有值均表示百分数
center=0.15 #中心偏离程度
leftCenter=0.01  #左高点与中心高点的差
rightCenter=0.01 #右高点与中心高点的差
bottomCenterMin=0.03  #最低点与最高点的差最小值
bottomCenterMax=0.20  #最低点与最高点的差最大值
bottomLeftRight=0.02  #两个最低点之间差的最大值
ratio1=-0.03    #隔一天的增减量最小值
ratio2=-0.02    #隔两天的增减量最小值
ratio3=-0.01   #隔三天的增减量最小值

wres=[]
for stockData in Data:
    try:
        # 计算开盘、收盘的平均价格
        data = []
        high = [] #最高价
        low = [] #最低价
        close = [] #收盘价
        for dat in stockData:
            high.append(float(dat[3]))
            low.append(float(dat[4]))
            close.append(float(dat[5]))
            if float(dat[3]) < 0:
                continue
            if mode == 1:
                data.append((float(dat[2]) + float(dat[5])) / 2)
                
        # 计算MACD
        macd,signal,hist = talib.MACD(np.array(close),fastperiod=12,slowperiod=26,signalperiod=9)
        macd[np.isnan(macd)] = 0
        signal[np.isnan(signal)] = 0
        hist[np.isnan(hist)] = 0
        # 计算ADX
        adx = talib.ADX(np.array(high), np.array(low), np.array(close))
        adx[np.isnan(adx)] = 0
        # 计算RSI
        rsi = talib.RSI(np.array(close))
        rsi[np.isnan(rsi)] = 0
        # 查找W形
        start=0
        end=start+min_len
        while(1):
            if start+min_len>=len(data)-1: break
            ts=(start,data[start])  # 左高点
            te = (end, data[end])   # 右高点
            centMax=max(data[start+2:end-1])
            th = (data[start+2:end-1].index(centMax)+start+2,centMax) # 中心高点
            leftMin=min(data[start+1:th[0]])
            tl=(data[start+1:th[0]].index(leftMin)+start+1,leftMin)  # 左低点
            rightMin=min(data[th[0]+1:end])
            tr=(data[th[0]+1:end].index(rightMin)+th[0]+1,rightMin)  # 右低点
            flag=False
            # 判断W形态各条件是否满足
            if abs(th[0]-(start+end)/2)<= (end-start)*center \
                    and (abs(ts[1]-th[1])/th[1]<leftCenter or ts[1]>th[1] and tl[1]<th[1])\
                    and (abs(te[1]-th[1])/th[1]<rightCenter or te[1]>th[1] and tr[1]<th[1]) \
                    and ts[1]>max(data[start+1:tl[0]+1]) \
                    and te[1]>max(data[tr[0]:end]) \
                    and (th[1]-tl[1])/tl[1]<bottomCenterMax*float(end-start)/min_len \
                    and (th[1]-tr[1])/tr[1]<bottomCenterMax*float(end-start)/min_len \
                    and (th[1]-tl[1])/tl[1]>bottomCenterMin*float(end-start)/min_len \
                    and (th[1]-tr[1]) / tr[1] > bottomCenterMin * float(end - start) / min_len \
                    and abs(tl[1]-tr[1])/min(tl[1],tr[1]) < bottomLeftRight*float(end-start)/min_len:
                flag=True
                for i in range(start+1,tl[0]+1):  #第一个下降区间
                    if (data[i]-data[i-1])/data[i]>ratio1*(-1) \
                            or i-start>=2 and (data[i]-data[i-2])/data[i]>ratio2*(-1) \
                            or i-start>=3 and (data[i]-data[i-3])/data[i]>ratio3*(-1):
                        flag=False
                        break
                for i in range(tl[0]+1,th[0]+1):  #第一个上升区间
                    if (data[i]-data[i-1])/data[i-1]<ratio1 \
                            or i-tl[0]>=2 and (data[i]-data[i-2])/data[i-2]<ratio2 \
                            or i-tl[0]>=3 and (data[i]-data[i-3])/data[i-3]<ratio3:
                        flag=False
                        break
                for i in range(th[0]+1,tr[0]+1):  #第二个下降区间
                    if (data[i]-data[i-1])/data[i]>ratio1*(-1) \
                            or i-th[0]>=2 and (data[i]-data[i-2])/data[i]>ratio2*(-1) \
                            or i-th[0]>=3 and (data[i]-data[i-3])/data[i]>ratio3*(-1):
                        flag=False
                        break
                for i in range(tr[0]+1,end):  #第二个上升区间
                    if (data[i]-data[i-1])/data[i-1]<ratio1 \
                            or i-tr[0]>=2 and (data[i]-data[i-2])/data[i-2]<ratio2 \
                            or i-tr[0]>=3 and (data[i]-data[i-3])/data[i-3]<ratio3:
                        flag=False
                        break
                if flag==True:  #找到了W形
                    # 计算训练样本标签并保存
                    yList=[0]
                    for i in range(end+1,min([end+11,len(data)])):
                        if (float(stockData[i][3])-float(stockData[end+1][2]))/float(stockData[end+1][2])>0.05:
                            yList=[1]
                            break
                        if (float(stockData[i][3])-float(stockData[end+1][2]))/float(stockData[end+1][2])<-0.05:
                            break
                    f = open("yList_1.txt", "a")
                    f.write(json.dumps(yList)+"\n")
                    f.close()
                    # 生成训练样本并表存
                    xList=[]
                    for i in range(start,start+max_len):
                        if i<=end:
                            xList.append([float(stockData[i][2]),float(stockData[i][3]),float(stockData[i][4]),float(stockData[i][5]),float(stockData[i][7]),
                                          float(macd[i]),float(signal[i]),float(hist[i]),float(adx[i]),float(rsi[i])])
                        else:
                            # 天数较少的样本后面补0
                            xList.append([0,0,0,0,0,0,0,0,0,0])
                    f = open("xList_1.txt", "a")
                    f.write(json.dumps(xList)+"\n")
                    f.close()
                    start=end
                    end=start+min_len
            if flag==False:
                if end-start>=max_len or end>=len(data)-1:
                    start=start+1
                    end=start+min_len
                else:
                    end=end+1
    except Exception as e:
        print(e)

