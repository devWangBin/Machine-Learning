# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 21:11:38 2018
@author: 93568
"""
import csv
import pprint
import math
import numbers
import decimal
import random
import json
import sys
from collections import defaultdict
from json import dumps, loads, JSONEncoder, JSONDecoder
import pickle
import copy
from operator import itemgetter 


if sys.version_info[0] != 2 or sys.version_info[1] < 6:
    print("This script requires Python version 2.6")
    sys.exit(1)

#将决策树以json写入磁盘并且读回的类
class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (list, dict, str, unicode, int, float, bool, type(None))):
            return JSONEncoder.default(self, obj)
        return {'_python_object': pickle.dumps(obj)}

def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(str(dct['_python_object']))
    return dct

myname = "201600130053WangBinE6"
pp = pprint.PrettyPrinter(indent=4)


#------------------------------------------
# decison tree
#------------------------------------------
# 决策树的类定义
class DecisionTree():

    def getKTiles(self, K, dataIdx,decisionAttrib,resultAttrib):
        
        ld = len(dataIdx)       
        dvals={}
        vals=[]
   
        for i in dataIdx:
              
            rVal = self.data[i][self.attribs[decisionAttrib]]

            if rVal not in dvals:
                dvals[rVal]=[i]
            else:
                dvals[rVal].append(i)
                
            vals.append(rVal)
                
        vals.sort()

        results = {}
        
        if ld <= 2:

            dkvals = dvals.keys()
            dkvals.sort()
            
            for vl in dkvals:
                
                if vl not in results:
                    results[vl] = {}
                    results[vl]['bin_records'] = []
                    results[vl]['bin_records_result_count'] = {}

                results[vl]['bin_records']=dvals[vl]
            
                for il in dvals[vl]:
                    
                    rVal = self.data[il][self.attribs[resultAttrib]] 
                    
                    if rVal in results[vl]['bin_records_result_count']:
                        results[vl]['bin_records_result_count'][rVal] += 1
                    else:
                        results[vl]['bin_records_result_count'][rVal] = 1
                        
                del dvals[vl]

        else:

            if (ld < 2*K):
                nk = ld/2
            else:
                nk = K
            
            gk = ld / nk
            rk = ld % nk
            previ = 0
            

            for i in range(1,nk+1):
                
                ki = i*gk-1
                if (i <= rk):
                    ki += i
                else:
                    ki += rk

                if vals[ki] not in results:
                    results[vals[ki]] = {}
                    results[vals[ki]]['bin_records'] = []
                    results[vals[ki]]['bin_records_result_count'] = {}
                    
                for vl in vals[previ:ki+1]:

                    if vl in dvals:

                        results[vals[ki]]['bin_records'].extend(dvals[vl])


                        for il in dvals[vl]:

                            rVal = self.data[il][self.attribs[resultAttrib]] 

                            if rVal in results[vals[ki]]['bin_records_result_count']:
                                results[vals[ki]]['bin_records_result_count'][rVal] += 1
                            else:
                                results[vals[ki]]['bin_records_result_count'][rVal] = 1

                        del dvals[vl]


                if len(results[vals[ki]]['bin_records']) == 0:
                    del results[vals[ki]]

                previ = ki+1
                        

        return results

    # 对训练集进行10次交叉验证
    # 学习决策树和调整参数
    # 选择K的大小 for Ktile
    # 选择修剪复杂成本来帮助修剪枝
    #
    # 从交叉验证结果返回最佳树             
    # 用于测量已被保留的测试数据的精度。 
   
    def learn(self, training_set, title):

        self.attribs = title.copy()
        self.attribType = {k:'numeric' for k in self.attribs.keys()}
        

        for v in training_set:
            if 'numeric' in self.attribType.values():
                for k in [ ak for ak in self.attribType.keys() if self.attribType[ak] == 'numeric' ]:
                    if ( not isinstance( v[self.attribs[k]], (int,float,long)) ):
                        self.attribType[k]='not_numeric'
            else:
                break


        cvResults={}
       
        bestNodeList = {}
        bestAvgAcc = float("-inf")
        bestBuf = None


      
        for k in [2,4,5,8]:

            for useGini in [True,False]:
                
                self.useGini = useGini

                cv_avgAcc=0

                self.kTile = k
     
                pbuf = "kTile=%d,costComplexityFactor=%.4f,Gini=%s" % ( self.kTile , float(self.ccf), self.useGini) 

                count=0.0
                for N in xrange(0, 10):

                    training_data = []
                    cv_set = []

                    training_data = [x for i, x in enumerate(training_set) if (i-N) % 10 != 0]
                    cv_set = [x for i, x in enumerate(training_set) if (i-N) % 10 == 0]

                    self.nodeList.clear()
                    del self.data[:]
                    self.data = training_data

                    self.train( training_data , title )

                    if self.pruneTree:
                      
                        self.prune(cv_set)
                  
                    cv_accuracy = self.getCVAccuracy(cv_set)

                    sbuf = "CrossValidation:N=%d->%s,training accuracy=%.4f" % (N, pbuf, cv_accuracy) 

                    print(sbuf)
                 
                    f = open(myname+"result-crossvalidation.txt", "ab+")
                    f.write(sbuf+" \n")
                    f.close()

                    cv_avgAcc += cv_accuracy
                    count +=1

                    del training_data[:]
                    del cv_set[:]
                         
            cv_avgAcc = float(cv_avgAcc)/float(count)

            cvResults[cv_avgAcc] = {
                'buf':pbuf
                } 
          
            if cv_avgAcc > bestAvgAcc:
                bestAvgAcc = cv_avgAcc
                bestNodeList.clear()
                bestNodeList = copy.deepcopy(self.nodeList)
                bestBuf = pbuf
        
        self.nodeList.clear()
        self.nodeList = bestNodeList
                           
        sbuf = "Best Tree from Cross Validation:%s,training accuracy=%.4f" % (bestBuf, bestAvgAcc ) 

        print(sbuf)
      
        f = open(myname+"result-crossvalidation.txt", "ab+")
        f.write(sbuf+" \n")
        f.close()

        f = open(myname+"result-crossvalidation.txt", "ab+")
        
        cvKs = cvResults.keys()
        cvKs.sort(reverse=True)
        for svr in cvKs[:N]:
        
            res=cvResults[svr]
            
            sbuf = "Best Training Results:%s,Average Training accuracy: %.4f" % ( res['buf'], svr)
            
            print(sbuf)
            f.write(sbuf+" \n")
            
        f.close()

    def saveToDiskAsJson(self,fileName):
        
        saveDict  = {}
        saveDict['attribs'] = self.attribs
        saveDict['attribType'] = self.attribType
        saveDict['nodeList'] = self.nodeList
		
        jSaveDict = dumps(saveDict, cls=PythonObjectEncoder)
        with open(fileName, 'w') as fp:
            json.dump(jSaveDict, fp)
            fp.close()
 
    def readTreeFromJsonFile(self, fileName):
        with open(fileName, 'r') as fp:
            jSaveDict = json.load(fp)
            saveDict = loads(jSaveDict, object_hook=as_python_object)
            self.attribs = saveDict['attribs']
            self.attribType = saveDict['attribType']
            self.nodeList = saveDict['nodeList']
             
    def __init__(self):
        self.attribs = []
        self.attribType = {}
        self.nodeList = {}
        self.data = []
        self.kTile = 15
        self.useGini = False
      
        self.useBinEntropy=False
        self.branchForEntropyGainOnly = False
        self.returnMajorityNotDefault = False
        self.pruneTree = True
        self.target = 'quality'
        self.ccf = 0
  
    def train ( self, training_data, title):
        
        rootNode = {
            'key':'root',
            'parent':None,
            'type' : 'root',
            'attribs':title.copy(), 
            'dataIdx': set(range(len(training_data))) , 
            'children': [], 
            'bin':None,
            'parentAttrib':None,
            'level':0,
            'chosenAttrib':None,
            'parentKey':None
            }

        self.nodeList['root']=rootNode
                    
        stack = []
        stack.append('root')

        while stack:

            nodeKey = stack.pop(0)

            node = self.nodeList[nodeKey]
                  
            lv = [self.data[i][self.attribs[self.target]] for i in node['dataIdx']]
            
            if len(lv):
                majorityValue = max(lv,key=lv.count)
                node['majorityValue']=majorityValue
          
            node['total']=len(node['dataIdx'])
			
            errCount=0
            for i in node['dataIdx']:
                if self.data[i][self.attribs[self.target]] != majorityValue:
                    errCount +=1

            node['error_for_majority']=errCount
  
            if ( len(node['attribs']) == 1):
                node['type']='leaf'
                continue
            else:
                
                etp1 = Entropy(self,node['dataIdx'],self.target,self.target)
                           
                if ( etp1 == 0 ):
                    node['type']='leaf'
                    continue
                else:
                
                    bestAttrib = ''
                    betp2=1.01
                    for bAttrib in set(set(node['attribs'])-set([self.target])):
                        
                        etp2 = Entropy(self,node['dataIdx'],self.target,bAttrib)
                        
                        if etp2 < betp2:
                            bept2 = etp2
                            bestAttrib = bAttrib
                                                                              
                    if node['type'] != 'root':
                        if self.branchForEntropyGainOnly:
                            if betp2 >= etp1:
                                node['type']='leaf'
                                continue

                 

                    if ( self.attribType[bestAttrib] == 'numeric' ):
                        aVResult = self.getKTiles(self.kTile, node['dataIdx'],bestAttrib,self.target)   
                        aVSet = set( aVResult.keys() )
                    else:
                        aVSet = set( self.data[i][self.attribs[bestAttrib]] for i in node['dataIdx'] )
                            
                    node['chosenAttrib']=bestAttrib
                    td=node['children']

                    for aVal in aVSet:
                     
                        if ( self.attribType[bestAttrib] == 'numeric' ):
                            vIdx = set( aVResult[aVal]['bin_records'] )                            
                        else:
                            vIdx = set( i for i in node['dataIdx'] if self.data[i][self.attribs[bestAttrib]] == aVal )
                        
                        vAttribs = node['attribs'].copy()
                        vAttribs.pop(bestAttrib)
                        
                        childNodeKey = nodeKey+"--"+bestAttrib+"--"+str(aVal)
                        td.append(childNodeKey)

                        childNode =  {
                            'type':'subTree',
                            'key':childNodeKey,
                            'attribs':vAttribs, 
                            'dataIdx': vIdx , 
                            'bin':aVal,
                            'children': [],
                            'parentAttrib':bestAttrib,
                            'parentKey': nodeKey,
                            'level': node['level']+1,
                            'chosenAttrib':None}

                        self.nodeList[childNodeKey]=childNode

                        stack.append(childNodeKey )
                         

    def prune(self,cv_set):
                      
        bestAccuracy = self.getCVAccuracy(cv_set)
        bestNodeList = copy.deepcopy(self.nodeList)
        
        self.pruneAllCostlierSubTrees(cv_set)
              
        cv_accuracy = self.getCVAccuracy(cv_set)
        if cv_accuracy > bestAccuracy:
            bestAccuracy = cv_accuracy
            bestNodeList.clear()
            bestNodeList = copy.deepcopy(self.nodeList)
       
        while len(self.nodeList.keys()) > 1:
          
            prStack = {}
         
            for nodeKey in self.nodeList:
                node = self.nodeList[nodeKey] 
                if node['type'] == 'leaf':
                    if node['level'] not in prStack:
                        prStack[node['level']] = []
                    prStack[node['level']].append(nodeKey)
           
            visited = {}
            minCcF = float('inf')
            maxLevel = max(prStack.keys())
            pruneNodeKey = None
           
            for level in reversed(xrange(1,maxLevel+1)):
              
                while prStack[level]:
                                    
                    node = self.nodeList[prStack[level].pop(0)]

                    if node['key'] in visited:
                        continue;

                    visited[node['key']]=1
                   
                    if node['parentKey']:
                        if node['level'] not in prStack:
                            prStack[node['level']] = []
                        prStack[node['level']].append(node['parentKey'])
                   
                    if node['type'] == 'leaf':
                        
                        node['errors'] = node['error_for_majority'] 
                        node['leaves'] = 1
                        node['ccf'] = float('inf')
                   
                    elif node['type'] != 'leaf':
                      
                        cErrors = 0.0
                        leaves =  0.0
                        
                        for ck in node['children']:
                            
                            cNode = self.nodeList[ck]
                            cErrors +=  cNode['errors']
                            leaves += float(cNode['leaves'])
                       
                        node['leaves'] = leaves
                        node['errors'] = cErrors                            
                        if float(leaves) != 1:
                            node['ccf'] =  ( float(node['error_for_majority']) - cErrors ) / float(leaves-1.0)
                        else:
                            node['ccf'] = float('-inf')
                       
                        if node['ccf'] <= minCcF:
                            minCcF = node['ccf']
                            pruneNodeKey = node['key']         
           
            if pruneNodeKey:
                self.pruneSubTree(pruneNodeKey)
                
            cv_accuracy = self.getCVAccuracy(cv_set)
            if cv_accuracy > bestAccuracy:
                bestAccuracy = cv_accuracy
                bestNodeList.clear()
                bestNodeList = copy.deepcopy(self.nodeList)
      
        self.nodeList.clear()
        self.nodeList = bestNodeList
        
   
    def pruneAllCostlierSubTrees(self,cv_set):
                
        bestAccuracy = float('-inf')
        bestNodeList = copy.deepcopy(self.nodeList)
        savedNodeList = copy.deepcopy(self.nodeList)

        for ccf in [0.0,0.001,0.01,0.05,0.1,0.2,0.4,0.6,0.8,1.0]:
                      
            self.nodeList.clear()
            self.nodeList = copy.deepcopy(savedNodeList)
         
            prStack = {}
                      
            for nodeKey in self.nodeList:
                node = self.nodeList[nodeKey] 
                if node['type'] == 'leaf':
                    if node['level'] not in prStack:
                        prStack[node['level']] = []
                    prStack[node['level']].append(nodeKey)

            visited = {}

            maxLevel = max(prStack.keys())

            for level in reversed(xrange(1,maxLevel+1)):

                while prStack[level]:
                 
                    node = self.nodeList[prStack[level].pop(0)]

                    if node['key'] in visited:
                        continue;

                    visited[node['key']]=1
                 
                    if node['parentKey']:
                        if node['level'] not in prStack:
                            prStack[node['level']] = []                       
                        prStack[node['level']].append(node['parentKey'])

                    if node['type'] == 'leaf':

                        node['errors'] = node['error_for_majority'] 
                        node['leaves'] = 1

                  
                    if node['type'] != 'leaf':
                     
                        cErrors = 0
                        leaves = 0

                        for ck in node['children']:

                            cNode = self.nodeList[ck]
                            cErrors +=  cNode['errors']
                            leaves += float(cNode['leaves'])

                        node['leaves'] = leaves
                     
                        if  node['error_for_majority']  > ( cErrors + ccf*(leaves-1) ):
                            node['errors'] = cErrors
                       
                        else:
                            self.pruneSubTree(node['key'])

            cv_accuracy = self.getCVAccuracy(cv_set)
            if cv_accuracy > bestAccuracy:
                bestAccuracy = cv_accuracy
                bestNodeList.clear()
                bestNodeList = copy.deepcopy(self.nodeList)
                self.ccf = ccf
   
        self.nodeList.clear()
        self.nodeList = bestNodeList

		
    def pruneSubTree(self,pruneNodeKey):

        node = self.nodeList[pruneNodeKey]
                
        node['errors']= node['error_for_majority']
        node['type']='leaf'
        node['chosenAttrib']=None
        node['leaves']=1
        node['ccf']=float('inf')
      
        pruneStack = node['children']

        node['children']=[]

        while pruneStack:

            prNodeKey = pruneStack.pop(0)
            prNode = self.nodeList[prNodeKey]


            if prNode['type'] != 'leaf':
                for ck in prNode['children']:
                    pruneStack.append(ck)

            del self.nodeList[prNode['key']]


    def getCVAccuracy(self,cv_set):

        cv_results = []
        for instance in cv_set:
            cv_result = self.classify( instance[:-1] )
            cv_results.append( cv_result == instance[-1])
            

        cv_accuracy = float(cv_results.count(True))/float(len(cv_results))

        return cv_accuracy
        

    def printTree(self):

        pStack = []
        pStack.append('root')

        while pStack:

            node = self.nodeList[pStack.pop()]
            
            idC = '\t' * node['level']
            if node['type']=='root':
                print("Root")
            else:
                print("%s%s=%s" % (idC,node['parentAttrib'],node['bin']) )
            
            if node['type'] == 'leaf':
                print("%s%s=%s" % (idC,self.target,node['majorityValue']) )

            for chk in node['children']:
                pStack.append(chk)


    def classify(self, test_instance):
        result = 0 # 基线版本总是分为0
 
        nodeKey = 'root'

        while nodeKey:
                      
            node = self.nodeList[nodeKey]


            if node['type'] == 'leaf':
                return node['majorityValue']

            attrib = node['chosenAttrib']
 

            tval = test_instance[self.attribs[attrib]]


            nodeKey=None
            if self.attribType[attrib] == 'numeric':

                prev=float("-inf")
                lastNK=None

                nkDs = {self.nodeList[cnK]['bin']:cnK for cnK in node['children']}
                nkKs = nkDs.keys()
                nkKs.sort()
                
                for binVal in nkKs:
                    nK = nkDs[binVal]
                    if prev < tval <= binVal:
                        nodeKey = nK
                        break
                    prev=binVal
                    lastNK=nK
 
                if self.returnMajorityNotDefault and not nodeKey:
                    return node['majorityValue']
   
            else:
                
                for nK,binVal in [(cnK,self.nodeList[cnK]['bin']) for cnK in node['children']]:   
                    if tval.lower() == binVal.lower():
                        nodeKey=nK
                        break
                    
                if self.returnMajorityNotDefault and not nodeKey:
                    return node['majorityValue'] 
             
        return result

    
#计算熵
def Entropy(tree,dataIdx,resultParam,decisionParam):

    decisionCount = {}

    r = tree.attribs[resultParam]
    d = tree.attribs[decisionParam]

    if ( tree.useBinEntropy and tree.attribType[decisionParam] == 'numeric' ):

        aVResult = tree.getKTiles(tree.kTile, dataIdx,decisionParam,resultParam)  

        aVSet = set( aVResult.keys() )

        for aVal in aVSet:

            decisionCount[aVal]={}
            decisionCount[aVal]['totalxx']=len(aVResult[aVal]['bin_records'])


            for rVal in aVResult[aVal]['bin_records_result_count'].keys():
                decisionCount[aVal][rVal]=aVResult[aVal]['bin_records_result_count'][rVal]     

    else:
        for i in dataIdx:

            e = tree.data[i]


            if e[d] in decisionCount:
                decisionCount[e[d]]['totalxx'] += 1
            else:
                decisionCount[e[d]] = {}
                decisionCount[e[d]]['totalxx'] = 1

                if e[r] in decisionCount[e[d]]:
                    decisionCount[e[d]][e[r]] += 1
                else:
                    decisionCount[e[d]][e[r]] = 1


    tot=len(dataIdx)
    en=float(0)

    if not tree.useGini:

        for dc in decisionCount:
            dct = decisionCount[dc]['totalxx']
            for rc in decisionCount[dc]:
                if rc != 'totalxx':
                    rct = decisionCount[dc][rc]
                    if resultParam == decisionParam:
                        en += -(rct * (math.log(rct,2)-math.log(tot,2)) )/tot
                    else:
                        en += -(dct*rct * (math.log(rct,2)-math.log(dct,2)) )/(dct*tot)

    else:

        # 计算GINI系数
        if ( resultParam == decisionParam ):

            en = 1.0

            for dc in decisionCount:
                dct = decisionCount[dc]['totalxx']
                for rc in decisionCount[dc]:
                    if rc != 'totalxx':
                        rct = decisionCount[dc][rc]
                        en = en-math.pow(float(rct)/float(tot),2)

        else:

            for dc in decisionCount:
                dct = decisionCount[dc]['totalxx']
                en += float(dct)/float(tot)
                for rc in decisionCount[dc]:
                    if rc != 'totalxx':
                        rct = decisionCount[dc][rc]
                        en = en-(float(dct)*math.pow(float(rct)/float(dct),2))/(tot)

    return en




def strtof(st):
    try:
        return float(st)
    except ValueError:
        return st
    

def common_val(lst):
    return max(set(lst),key=lst.count)


def fill_common_val(data,cVal,i,j):
       data[i][j] = cVal
        

def run_decision_tree(useSavedDecisonTree,fileName):

    # 加载数据集
    with open("ex6Data.csv") as f:
        reader = csv.reader(f, delimiter=",")
        # 读取标题
        title = { e:i for i,e in enumerate(next(reader))}
        # 读取所有的数据
        data = [map(strtof,line) for line in reader]

    dln = len(data)
    print ("Number of records: %d" % dln)


    if not fileName:
        fileName = "my_decision_tree.json"

    # PREPROCESS
    #
    # 得到普遍的数据并对空的进行填充
    cVals = [common_val([i[j] for i in data]) for j in range(len(title)) ]
   
    [ fill_common_val(data,cVals[j],i,j) for i in range(len(data)) for j in range(len(title)) if ( isinstance(data[i][j],str) and not data[i][j] ) ]

    # 划分数据集
    K = 10
    training_set = [x for i, x in enumerate(data) if i % K != 9]
    test_set = [x for i, x in enumerate(data) if i % K == 9]
    
    tree = DecisionTree()

    if  useSavedDecisonTree :
    
        print("Reading the decision tree from disk based json file")
        # 读取已经存储的决策数并用它进行分类测试
        tree.readTreeFromJsonFile(fileName)       

    else:     
        
        tree.learn( training_set, title)
      
        print("Saving the decision tree to disk as json")

        tree.saveToDiskAsJson(fileName)

    results = []
    for instance in test_set:
        result = tree.classify( instance[:-1] )
        results.append( result == instance[-1])
  
    accuracy = float(results.count(True))/float(len(results))
    print ("accuracy: %.4f" % accuracy)       
    
    f = open(myname+"result.txt", "w")
    f.write("accuracy: %.4f" % accuracy)
    f.close()


     
if __name__ == "__main__":

    if len(sys.argv) > 1:
        if len(sys.argv) > 2 :
            run_decision_tree(True,sys.argv[2])
        else:
            run_decision_tree(True,None)
    else:
        run_decision_tree(False,None)
		