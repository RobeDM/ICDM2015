#    Copyright (C) 2015  Roberto Diaz Morales
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


import csv
import numpy as np
import re
from collections import Counter,defaultdict
import os
import inspect
import sys
import sklearn
from sklearn import cross_validation
import pickle
import xgboost as xgb



#######################################################################################
# THIS FUNCTION PARSES THE FILES WITH THE INFORMATION ABOUT DEVICES AND COOKIES       #
# AND CREATES LISTS WITH THE IDENTIFIERS OF THE CATEGORICAL FEATURES.                 #
# THE INDEX OF THE LIST WILL BE USED AS THE VALUE OF THE FEATURE IN THE NUMPY MATRICES#
#######################################################################################

def GetIdentifiers(trainfile,testfile,cookiefile):

    DeviceList=list()
    CookieList=list()
    HandleList=list()
    DevTypeList=list()
    DevOsList=list()
    ComputerOsList=list()
    ComputerVList=list()
    CountryList=list()
    annC1List=list()
    annC2List=list()
    
    
    with open(trainfile,'rb') as csvfile:
        spamreader=csv.reader(csvfile,delimiter=',')
        spamreader.next()
        for row in spamreader:
            HandleList.append(row[0])
            DeviceList.append(row[1])
            DevTypeList.append(row[2])
            DevOsList.append(row[3])
            CountryList.append(row[4])
            annC1List.append(row[6])
            annC2List.append(row[7])

    DeviceList=list(set(DeviceList))
    CookieList=list(set(CookieList))
    HandleList=list(set(HandleList))
    DevTypeList=list(set(DevTypeList))
    DevOsList=list(set(DevOsList))
    CountryList=list(set(CountryList))
    annC1List=list(set(annC1List))
    annC2List=list(set(annC2List))


    with open(testfile,'rb') as csvfile:
        spamreader=csv.reader(csvfile,delimiter=',')
        spamreader.next()
        for row in spamreader:
            HandleList.append(row[0])
            DeviceList.append(row[1])
            DevTypeList.append(row[2])
            DevOsList.append(row[3])
            CountryList.append(row[4])
            annC1List.append(row[6])
            annC2List.append(row[7])            

    DeviceList=list(set(DeviceList))
    CookieList=list(set(CookieList))
    HandleList=list(set(HandleList))
    DevTypeList=list(set(DevTypeList))
    DevOsList=list(set(DevOsList))
    CountryList=list(set(CountryList))
    annC1List=list(set(annC1List))
    annC2List=list(set(annC2List))

    with open(cookiefile,'rb') as csvfile:
        spamreader=csv.reader(csvfile,delimiter=',')
        spamreader.next()
        for row in spamreader:
            HandleList.append(row[0])
            CookieList.append(row[1])
            ComputerOsList.append(row[2])
            ComputerVList.append(row[3])
            CountryList.append(row[4])
            annC1List.append(row[6])
            annC2List.append(row[7])

    DeviceList=list(set(DeviceList))
    CookieList=list(set(CookieList))
    HandleList=list(set(HandleList))
    DevTypeList=list(set(DevTypeList))
    DevOsList=list(set(DevOsList))
    ComputerOsList=list(set(ComputerOsList))
    ComputerVList=list(set(ComputerVList))
    CountryList=list(set(CountryList))
    annC1List=list(set(annC1List))
    annC2List=list(set(annC2List))

    return (DeviceList, CookieList, HandleList, DevTypeList, DevOsList,ComputerOsList,ComputerVList,CountryList,annC1List,annC2List)


############################################################################################################
# THIS FUNCTION RECEIVES A LIST AND CREATES A DICTIONARY TO GET THE INDEX WHEN THE VALUE IS GIVEN AS A KEY #
############################################################################################################

def list2Dict(lista):
    newDict=dict()
    for i in range(len(lista)):
        newDict[lista[i]]=i
    return newDict


##############################################################################
# THIS FUNCTION CREATES A NUMPY MATRIX WITH THE INFORMATION OF A DEVICE FILE #
##############################################################################
    
def loadDevices(trainfile,DictHandle,DictDevice,DictDevType,DictDevOs,DictCountry,DictAnnC1,DictAnnC2):

    NumRows = 0
    with open(trainfile,'rb') as csvfile:
        spamreader=csv.reader(csvfile,delimiter=',')
        spamreader.next()
        for row in spamreader:
            NumRows = NumRows + 1

    XDevices = np.zeros((NumRows,11))
    
    NumRows = 0
    with open(trainfile,'rb') as csvfile:
        spamreader=csv.reader(csvfile,delimiter=',')
        spamreader.next()
        for row in spamreader:
            XDevices[NumRows,0]=DictHandle[row[0]]
            XDevices[NumRows,1]=DictDevice[row[1]]
            XDevices[NumRows,2]=DictDevType[row[2]]
            XDevices[NumRows,3]=DictDevOs[row[3]]
            XDevices[NumRows,4]=DictCountry[row[4]]
            XDevices[NumRows,5]=np.float_(row[5])
            XDevices[NumRows,6]=DictAnnC1[row[6]]
            XDevices[NumRows,7]=DictAnnC2[row[7]]
            XDevices[NumRows,8]=np.float_(row[8])
            XDevices[NumRows,9]=np.float_(row[9])
            XDevices[NumRows,10]=np.float_(row[10])
            
            NumRows = NumRows + 1

    return XDevices


##############################################################################
# THIS FUNCTION CREATES A NUMPY MATRIX WITH THE INFORMATION OF A COOKIE FILE #
##############################################################################
    
def loadCookies(cookiefile,DictHandle,DictCookie,DictComputerOs,DictComputerV,DictCountry,DictAnnC1,DictAnnC2):
    
    maxindex=np.int(np.max(DictCookie.values()))
    
    XCookies = np.zeros((maxindex+1,11))
    
    with open(cookiefile,'rb') as csvfile:
        spamreader=csv.reader(csvfile,delimiter=',')
        spamreader.next()
        for row in spamreader:
            fila=np.int(DictCookie[row[1]])
            XCookies[fila,0]=DictHandle[row[0]]
            XCookies[fila,1]=DictCookie[row[1]]
            XCookies[fila,2]=DictComputerOs[row[2]]
            XCookies[fila,3]=DictComputerV[row[3]]
            XCookies[fila,4]=DictCountry[row[4]]
            XCookies[fila,5]=np.float_(row[5])
            XCookies[fila,6]=DictAnnC1[row[6]]
            XCookies[fila,7]=DictAnnC2[row[7]]
            XCookies[fila,8]=np.float_(row[8])
            XCookies[fila,9]=np.float_(row[9])
            XCookies[fila,10]=np.float_(row[10])
            
    return XCookies    

####################################################################################################
# THIS FUNCTION CREATES A DICTIONARY WHERE THE KEYS ARE THE IP ADDRESSES OF THE IP AGGREGATED FILE #
# AND THE VALUE A NUMPY ARRAY WITH ITS INFORMATION.                                                #
####################################################################################################

def loadIPAGG(ipaggfile):
    
    XIPS=dict()
    
    with open(ipaggfile,'rb') as csvfile:
        spamreader=csv.reader(csvfile,delimiter=',')
        spamreader.next()
        for row in spamreader:

            datoIP=np.zeros(5)
            datoIP[0]=np.float_(row[1])
            datoIP[1]=np.float_(row[2])
            datoIP[2]=np.float_(row[3])
            datoIP[3]=np.float_(row[4])
            datoIP[4]=np.float_(row[5])

            XIPS[row[0]]=datoIP            

    return XIPS

#####################################################################
# THIS FUNCTION CREATES A DICTIONARY WHERE THE KEYS ARE THE DEVICES #
# AND THE VALUE DICTIONARY OF THE PROPERTIES AND ITS INFORMATION    #
#####################################################################
   
def loadPROPS(fileprops,DictDevice,DictCookie):

    DevProps=dict()
    
    with open(fileprops) as fp:
        fp.readline()
        
        for line in fp:
            
            matchObj = re.match( r'([a-zA-Z0-9_]*),([0-9\-]*),{([(a-zA-Z0-9.(),\-_]*)}', line, flags=0)            
 
            if(matchObj.group(2)=='0'):
                props = re.findall(r'\((.*?)\)',matchObj.group(3))
                ValProps=dict()
                for prop in props:
                    propV = prop.split(',')
                    ValProps[propV[0]]=np.float_(propV[1])
                Devic=DictDevice.get(matchObj.group(1),-1)
                if Devic>-1:
                    DevProps[Devic]=ValProps

    return DevProps

#################################################################################################
# THIS FUNCTION CREATES:                                                                        #
# A DICTIONARY WHERE THE KEYS ARE THE DEVICES OF THE TRAINING SET AND THE VALUES THEIR COOKIES  #
# A DICTIONARY WHERE THE KEYS ARE THE COOKIES AND THE VALUES OTHER COOKIES WITH THE SAME HANDLE #
# A DICTIONARY WHERE THE KEYS ARE THE COOKIES AND THE VALUES THE DEVICES WITH THE SAME HANDLE   #
#################################################################################################

def creatingLabels(XDevices,XCookies,DictHandle):

    HDC=dict()
    unknown = DictHandle['-1']
    Handles=np.unique(XCookies[:,0])
    for i in range(len(Handles)):
        if Handles[i] != unknown:
            HDC[Handles[i]]=dict()
            HDC[Handles[i]]['Devices']=set()
            HDC[Handles[i]]['Cookies']=set()

    (NDevices,NDim)=XDevices.shape

    for i in range(NDevices):
        HDC[XDevices[i,0]]['Devices'].add(XDevices[i,1])

    (NCookies,NDim)=XCookies.shape

    for i in range(NCookies):
        if XCookies[i,0] != unknown:
            mdic=HDC.get(XCookies[i,0])
            mdic['Cookies'].add(XCookies[i,1])


    Labels=dict()
    Groups = dict()
    WhosDevice=dict()

    for k,v in HDC.iteritems():
        for dev in v['Devices']:
            Labels[dev]=v['Cookies']
        for coo in v['Cookies']:
            Groups[coo]=v['Cookies']
            WhosDevice[coo]=v['Devices']

    for i in range(NCookies):
        if XCookies[i,0] == unknown:
            name=XCookies[i,1]
            setcoo=set()
            setcoo.add(name)
            Groups[name]=setcoo

    return (Labels,Groups,WhosDevice)

############################################################################
# THIS FUNCTION EVALUATES THE F05 SCORE ON THE RESULTS OF A VALIDATION SET #
############################################################################
 
def calculateF05(Results,Target):

    BetaQ=0.5*0.5

    F05=list()

    for k in Results.keys():
        pos=Results[k]
        tla=Target[k]

        tp=np.float_(len(pos & tla))
        fp=np.float_(len(pos)-tp)
        fn=np.float_(len(tla)-tp)
        p=tp/(tp+fp)
        r=tp/(tp+fn)
        if p*r>0.0:
            f=(1.0+BetaQ)*p*r/(BetaQ*p+r)
        else:
            f=0.0
        F05.append(f)
    return np.mean(F05)

#################################################
# THIS FUNCTION CREATES THE DATA STRUCTURES TO: #
# FIND THE IP ADDRESSES OF EVERY DEVICE         #
# FIND THE IP ADDRESSES OF EVERY COOKIE         #
# FIND THE DEVICES OF EVERY IP ADDRESS          #
# FINC THE COOKIES OF EVERY IP ADDRESS          #
#################################################
   
def loadIPS(ipfile,DictDevice,DictCookie,XIPS,Groups):
   
    DeviceIPS=dict()
    CookieIPS=dict()
    IPDev=defaultdict(set)
    IPCoo=defaultdict(set)
   
    with open(ipfile) as fp:
        fp.readline()

        for line in fp:
            matchObj = re.match( r'([a-zA-Z0-9_]*),([0-9\-]*),{([(a-zA-Z0-9(),\-_]*)}', line, flags = 0)            
            ips = re.findall(r'(\w*,\w*,\w*,\w*,\w*,\w*,\w*)',matchObj.group(3))

            ValIPS=dict()
            for ip in ips:
                Indiv = ip.split(',')  
                arr=np.zeros(11)
                arr[0]=np.float_(Indiv[1])
                arr[1]=np.float_(Indiv[2])
                arr[2]=np.float_(Indiv[3])
                arr[3]=np.float_(Indiv[4])
                arr[4]=np.float_(Indiv[5])
                arr[5]=np.float_(Indiv[6])   
                dIP=XIPS[Indiv[0]]
                arr[6]=np.float_(dIP[0])
                arr[7]=np.float_(dIP[1])
                arr[8]=np.float_(dIP[2])
                arr[9]=np.float_(dIP[3])
                arr[10]=np.float_(dIP[4])

                ValIPS[Indiv[0]]=arr                           
                    
            if(matchObj.group(2)=='0'):            
                Device=DictDevice.get(matchObj.group(1),-1)
                if Device>-1:
                    DeviceIPS[Device]=ValIPS
                    for k in ValIPS.keys():
                        IPDev[k].add(Device)
                else:
                    DeviceIPS[matchObj.group(1)]=ValIPS
                    for k in ValIPS.keys():
                        IPDev[k].add(matchObj.group(1))
                    

            else:
                Cookie=DictCookie[matchObj.group(1)]
                CookieIPS[Cookie]=ValIPS
                for k in ValIPS.keys():
                    IPCoo[k].add(Cookie)



    for k,v in Groups.iteritems():
        if len(v)>1:
            for cook1 in v:
                for cook2 in v:
                    if cook1 != cook2:
                        d1=CookieIPS[cook1]
                        d2=CookieIPS[cook2]
                        for n1,n2 in d1.iteritems():
                            if n1 not in d2.keys():
                                d2[n1]=n2
                                IPCoo[n1].add(cook2)

    return (IPDev,IPCoo,DeviceIPS,CookieIPS)

################################################################################
# THIS FUNCTION FOR A GIVEN DEVICE CREATES:                                    #
# A SET OF COOKIES WITH KNOWN HANDLE THAT SHARE IP ADDRESSES WITH THE DEVICE   #
# A SET OF COOKIES WITH UNKNOWN HANDLE THAT SHARE IP ADDRESSES WITH THE DEVICE #
################################################################################

def fullCandidates(device,XDevices,XCookies,IPDev,IPCoo,DeviceIPS,DictHandle):

    CandidatesKnown=dict()
    CandidatesUnknown=dict()

    candidatestotalKnown=set()
    candidatestotalUnknown=set()

    Unknown = DictHandle['-1']

    ips=DeviceIPS[device].keys()
    
    for ip in ips:
        if(len(IPDev.get(ip,set()))<=30):
            candidates=IPCoo[ip]
            for candidate in candidates:
                if(XCookies[np.int(candidate),0] != Unknown):
                    candidatestotalKnown.add(candidate)
                else:
                    candidatestotalUnknown.add(candidate)

    if (len(candidatestotalKnown)==0):
        for ip in ips:
            candidates=IPCoo[ip]
            for candidate in candidates:
                if(XCookies[np.int(candidate),0] != Unknown):
                    candidatestotalKnown.add(candidate)
                else:
                    candidatestotalUnknown.add(candidate)


    CandidatesKnown[device]=candidatestotalKnown
    CandidatesUnknown[device]=candidatestotalUnknown
        
    return (CandidatesKnown,CandidatesUnknown)

###############################################################################
# THIS FUNCTION CREATES THE INITIAL SELECTION OF CANDIDATES FOR EVERY DEVICE  #
###############################################################################
                
def selectCandidates(XDevices,XCookies,IPDev,IPCoo,DeviceIPS,CookieIPS,DictHandle):

    devices = np.unique(XDevices[:,1])
    Candidates=dict()
    
    Unknown=DictHandle['-1']
    
    for i in range(len(devices)):
        device = devices[i]

        candidatestotal=set()
        ips=DeviceIPS[device].keys()
        for ip in ips:
            if(len(IPDev.get(ip,set()))<=10 and len(IPCoo.get(ip,set()))<=20):
                candidates=IPCoo[ip]
                for candidate in candidates:
                    if(XCookies[np.int(candidate),0] != Unknown):
                        candidatestotal.add(candidate)
    
        if len(candidatestotal)==0:
            for ip in ips:
                if(len(IPDev.get(ip,set()))<=25 and len(IPCoo.get(ip,set()))<=50):
                    candidates=IPCoo[ip]
                    for candidate in candidates:
                        if(XCookies[np.int(candidate),0] != Unknown):
                            candidatestotal.add(candidate)
    
    
        if len(candidatestotal)==0:
            for ip in ips:
                candidates=IPCoo[ip]
                for candidate in candidates:
                    if(XCookies[np.int(candidate),0] != Unknown):
                        candidatestotal.add(candidate)

        if len(candidatestotal)==0:
            for ip in ips:
                candidates=IPCoo[ip]
                for candidate in candidates:
                    candidatestotal.add(candidate)
    


        Candidates[device]=candidatestotal
        
    return Candidates


###########################################
# THIS CREATES A THE TRAINING OR TEST SET #
###########################################


def createDataSet(Candidates,XDevice,XCookies,DeviceIPS,CookieIPS,IPDev,IPCoo,Groups,WhosDevice,DevProps):

    OriginalIndex=dict()
    numdifs=0
    numpatterns=0
    for k,v in Candidates.iteritems():
        numpatterns=numpatterns+len(v)


    Added=0
    for k,v in Candidates.iteritems():
        Device=XDevice[XDevice[:,1]==k,np.array([2,3,4,5,6,7,8,9,10])]

        IndivIndex=dict()

        setk=set()
        setk.add(k)
        setdevips=set(DeviceIPS.get(k,dict()).keys())
        setdevpro=set(DevProps.get(k,dict()).keys())

        for coo in v:

            Cookie=XCookies[np.int(coo),np.array([2,3,4,5,6,7,8,9,10])]
    
            row=np.concatenate((Device,Cookie))


            setcooips=set(CookieIPS.get(coo,dict()).keys())
    
            PROPS=setdevpro
            mipro=PROPS
       

            IPS=(setdevips & setcooips)
            miips=set()
            for ip in IPS:
                if(len(IPDev.get(ip,set()))<=10 and len(IPCoo.get(ip,set()))<=20):
                    miips.add(ip)
            if len(miips)==0:
                for ip in IPS:
                    miips.add(ip)

            OtherDevices=set(WhosDevice.get(coo,set()))-setk

            devp=set()
            devi=set()

            for odev in OtherDevices:
                devp=devp | set(DevProps.get(odev,dict().keys()))
                devi=devi | set(DeviceIPS.get(odev,dict().keys()))


            intersec=np.float_(len(devp & setdevpro))
            interseci=np.float_(len(devi & setdevips))


            if intersec>0:
                intersec=intersec/np.float_(len(setdevpro))

            if interseci>0:
                intersec=intersec/np.float_(len(setdevips))


            row=np.concatenate((row,np.array([np.float_(len(OtherDevices))])))
            row=np.concatenate((row,np.array([np.float_(intersec)])))

            row=np.concatenate((row,np.array([np.float_(interseci)])))


            row=np.concatenate((row,np.array([np.float_(len(IPS))])))
            row=np.concatenate((row,np.array([np.float_(len(setdevips))])))
            row=np.concatenate((row,np.array([np.float_(len(setcooips))])))

            row=np.concatenate((row,np.array([np.float_(len(PROPS))])))
            row=np.concatenate((row,np.array([np.float_(len(setdevpro))])))


            row=np.concatenate((row,np.array([np.float_(len(Groups.get(coo,set())))])))
            row=np.concatenate((row,np.array([np.float_(len(Groups.get(coo,set()) & v))])))            
                            
            row=np.concatenate((row,np.array([np.float_(len(miips))])))


            iprow=np.zeros(22)
            niprows=0
            for ip in miips:                
                iprow=iprow+np.concatenate((DeviceIPS[k][ip].reshape(-1),CookieIPS[coo][ip].reshape(-1)))
                niprows=niprows+1

            if niprows>0:
                meaniprows=iprow/np.float_(niprows)
            else:
                meaniprows=iprow


            row=np.concatenate((row.reshape(-1),iprow.reshape(-1)))
            row=np.concatenate((row.reshape(-1),meaniprows.reshape(-1)))
            row=np.concatenate((row.reshape(-1),(iprow[0:6]-iprow[11:-5]).reshape(-1)))
                

            if Added==0:
                XTR=np.zeros((numpatterns,len(row)))

            IndivIndex[coo]=Added

            XTR[Added,:]=row

            Added=Added+1
        OriginalIndex[k]=IndivIndex
    return (XTR,OriginalIndex)

#####################################################
# THIS CREATES A THE LABELS FOR SUPERVISED LEARNING #
#####################################################

def createTrainingLabels(Candidates,Labels):

    numpatterns=0
    
    for k,v in Candidates.iteritems():
        numpatterns=numpatterns+len(v)

    YTR=np.zeros(numpatterns)

    Added=0
    for k,v in Candidates.iteritems():
        for coo in v:
            if(coo in Labels[k]):
                YTR[Added]=1.0
            Added=Added+1

    return YTR

       
######################################################
# THIS FINCTION SELECTS THE COOKIES FOR EVERY DEVICE #
# GIVEN THE PREDICTIONS OF THE CLASSIFIER            #
######################################################

def bestSelection(predictions, OriginalIndex, values,Groups):

    result=dict()

    threshold=dict()

    for k,v in OriginalIndex.iteritems():

        cook=set()
        maxval=0.0
        cookies=v.keys()

        scores=np.zeros(len(cookies))

        for i in range(len(cookies)):
            scores[i]=predictions[v[cookies[i]]]


        Orden=sorted(range(len(scores)),key=lambda x:-scores[x])

        if len(cookies)>0:
            if Groups.get(cookies[Orden[0]],-100) != -100:
                maxval=scores[Orden[0]]
                cook= (cook | Groups[cookies[Orden[0]]])

        if (maxval<0.9):
            for i in range(len(values)):
                if (i<= len(cook)):
                    if (i<len(cookies) and (i<len(values))) :
                        tam1=len(Groups.get(cookies[Orden[0]],set()))
                        tam2=len(Groups.get(cookies[Orden[i]],set()))
                        if (tam1>1 & tam2==1):
                            if(scores[Orden[i]]>maxval*(values[i]-0.15)):
                                cook= (cook | Groups.get(cookies[Orden[i]],set()))
                        elif (tam1>1 & tam2>1):
                            if(scores[Orden[i]]>maxval*(values[i]+0.1)):
                                cook= (cook | Groups.get(cookies[Orden[i]],set()))
                        elif (tam1==1 & tam2==1):
                            if(scores[Orden[i]]>maxval*(values[i])):
                                cook= (cook | Groups.get(cookies[Orden[i]],set()))


        result[k]=cook
        threshold[k]=maxval
    return (result,threshold)

#####################################################
# THIS FUNCTION TRAINS THE CLASSIFIER USING XGBOOST #
#####################################################

def trainXGBoost(xtr,ytr,rounds,eta,xtst,ytst):
    xgmat = xgb.DMatrix( xtr, label=ytr)
    xgmat2 = xgb.DMatrix( xtst, label=ytst)
    param = {}
    param['eta'] = eta
    param['max_depth'] = 10
    param['subsample'] = 1.0
    param['nthread'] = 12
    param['min_child_weight']=4
    param['gamma']=5.0
    param['colsample_bytree']=1.0
    param['silent']=1
    param['objective'] = 'binary:logistic'
    param['eval_metric']='error'
    watchlist = [ (xgmat,'train') ,(xgmat2,'test')]
    num_round = rounds
    bst = xgb.train( param, xgmat, num_round, watchlist );
    return bst

#######################################
# THIS FUNCTION MAKES THE PREDICTIONS #
#######################################

def predictXGBoost(X,bst):
    xgmat = xgb.DMatrix( X)
    return bst.predict( xgmat )

#########################################################################
# THIS FUNCTION TRAINS THE ALGORITHM USING 8 BAGGERS AND AVERAGING THEM #
#########################################################################

def FullTraining(YTR,XTR,XTST,OriginalIndexTR,OriginalIndexTST,DevicesTrain,Groups,Labels):
    NFOLDS=8

    skf = sklearn.cross_validation.KFold(len(OriginalIndexTR.keys()),n_folds=NFOLDS,random_state=0)

    resultadosVal=np.zeros(len(YTR))


    (tamTST,dTST)=XTST.shape
    resultadosTST=np.zeros(tamTST)


    classifiers=list()

    iteration=0
    for (train,test) in skf:
        
        iteration=iteration+1
        Originaltmp=dict()
        print "Training Bagger ",iteration, "of", NFOLDS


        trainind=list()
        testind=list()
        traindev=list()
        testdev=list()

        for i in train:
             devtr=DevicesTrain[i,1]
             traindev.append(devtr)
             trainind.extend(OriginalIndexTR[devtr].values())

        for i in test:
            devtr=DevicesTrain[i,1]
            testdev.append(devtr)
            testind.extend(OriginalIndexTR[devtr].values())
            Originaltmp[devtr]=OriginalIndexTR[devtr]
 
        trainind=np.array(trainind)
        testind=np.array(testind)

        XvalTR=XTR[trainind,:]
        XvalTST=XTR[testind,:]
    
        YvalTR=YTR[trainind]
        YvalTST=YTR[testind]


        bst=trainXGBoost(XvalTR,YvalTR,200,0.10,XvalTST,YvalTST)

        classifiers.append((bst,traindev,testdev))

        pTT=predictXGBoost(XvalTR,bst)
        pTR=predictXGBoost(XvalTST,bst)

        resultadosVal[testind]=pTR

        (validat,thTR)=bestSelection(resultadosVal, Originaltmp, np.array([1.0]),Groups)

        pTST=predictXGBoost(XTST,bst)


        resultadosTST=resultadosTST+pTST


    resultadosTST=resultadosTST/np.float_(NFOLDS)
    return(resultadosVal,resultadosTST, OriginalIndexTR,OriginalIndexTST, classifiers)

###############################################################################################
# THIS FUNCTION LOOKS FOR DEVICES WHOSE BEST CANDIDATE SCORES LESS THAN 0.05,                 #
# CREATES A NEW SET OF CANDIDATES CONTAINING EVERY COOKIE THAT SHARES AN IP ADDRESS WITH HIM, #
# SCORES THEM WITH XGBOOST AND SELECT THE CANDIDATES FOR THE SUBMISSION                       #
###############################################################################################

def PostAnalysisTrain(validat,thTR,classifiers,DevicesTrain,Cookies,DeviceIPS,CookieIPS,IPDev,IPCoo,Groups,WhosDevice,DevProperties,DictHandle,Labels):

    itn=0
    for k,v in validat.iteritems():
        itn=itn+1
        if thTR[k]<0.05:
            (fcandK,fcandU)=fullCandidates(k,DevicesTrain,Cookies,IPDev,IPCoo,DeviceIPS,DictHandle)
         
            validatTHK=dict()
            thTHK=dict()
            if(len(fcandK[k])>0):
                (XTHK,OriginalIndexTHK)=createDataSet(fcandK,DevicesTrain,Cookies,DeviceIPS,CookieIPS,IPDev,IPCoo,Groups,WhosDevice,DevProperties)
                YTHK=createTrainingLabels(fcandK,Labels)
                estimK=np.zeros(len(YTHK))

                for (classifier,traindev,testdev) in classifiers:
                    if k in testdev:
                        estimK=predictXGBoost(XTHK,classifier)

                (validatTHK,thTHK)=bestSelection(estimK, OriginalIndexTHK, np.array([1.0,0.9]),Groups)
        
            validatTHU=dict()
            thTHU=dict()
            if(len(fcandU[k])>0):
                (XTHU,OriginalIndexTHU)=createDataSet(fcandU,DevicesTrain,Cookies,DeviceIPS,CookieIPS,IPDev,IPCoo,Groups,WhosDevice,DevProperties)
                YTHU=createTrainingLabels(fcandU,Labels)
                estimU=np.zeros(len(YTHU))

                for (classifier,traindev,testdev) in classifiers:
                    if k in testdev:
                        estimU=predictXGBoost(XTHU,classifier)

                (validatTHU,thTHU)=bestSelection(estimU, OriginalIndexTHU, np.array([1.0,0.9]),Groups)

            if len(validatTHK)>0:
                if len(validatTHU)>0:
                    if(thTHU[k]>(thTHK[k]+0.7)):
                        validat[k]=validatTHU[k]
                        thTR[k]=thTHU[k]
                    else:
                        if thTR[k]<=0.025:
                            validat[k]=validatTHK[k]
                            thTR[k]=thTHK[k]
                        else:
                            if thTR[k]+0.3<thTHK[k]:
                                validat[k]=validatTHK[k]
                                thTR[k]=thTHK[k]
                else:
                    if thTR[k]<=0.025:
                        validat[k]=validatTHK[k]
                        thTR[k]=thTHK[k]
                    else:
                        if thTR[k]+0.3<thTHK[k]:
                            validat[k]=validatTHK[k]
                            thTR[k]=thTHK[k]
            else:
                validat[k]=validatTHU[k]
                thTR[k]=thTHU[k]

    return(validat,thTR)

########################################################################
# THIS FUNCTION RETURNS THE DEVICES THAT SHARES IPS WITH ONLY 1 COOKIE #
########################################################################

def uniqueCandidates(XDevices,XCookies,IPCoo,DeviceIPS,DictHandle,OtherCookies):

    UniqueCandidates=dict()

    devices=np.unique(XDevices[:,1])

    numUnique=0

    Unknown=DictHandle['-1']
    for i in range(len(devices)):

        device=devices[i]

        candidatestotal=set()

        ips=DeviceIPS[device].keys()

        for ip in ips:
            candidates=IPCoo[ip]
            for candidate in candidates:
                if(XCookies[np.int(candidate),0] != Unknown):
                    candidatestotal.add(candidate)

        if len(candidatestotal)==0:
            for ip in ips:
                candidates=IPCoo[ip]
                candidatestotal=(candidatestotal | candidates)

        finallist=set()
        for candidate in candidatestotal:
            finallist=(finallist | OtherCookies[candidate])

        if OtherCookies[min(finallist)]==finallist:
            UniqueCandidates[device]=finallist
            numUnique=numUnique+1
        
    return UniqueCandidates

#################################################################################
# THIS FUNCTION RETURNS THE DEVICES WHOSE BEST CANDIDATE SCORES HIGHER THAN 0.4 #
# AND THE SECOND CANDIDATE SCORES LESS THAN 0.05                                #
#################################################################################

def mostProbable(predictions, OriginalIndex, Groups):
    
    probCandidates=dict()

    for k,v in OriginalIndex.iteritems():

        cookies=v.keys()
        scores=np.zeros(len(cookies))
        
        for i in range(len(cookies)):
            scores[i]=predictions[v[cookies[i]]]
        
        Orden=sorted(range(len(scores)),key=lambda x:-scores[x])

        ValorMax=-1
        cook=set()
        if len(cookies)>0:
            if Groups.get(cookies[Orden[0]],-100) != -100:
                cook= (cook | Groups[cookies[Orden[0]]])
                ValorMax=scores[Orden[0]]
    
        Segun=-1
        Terminado='NO'

        for i in range(len(cookies)):
            if i>0:    
                if Terminado=='NO':
                    if (cookies[Orden[i]] not in cook):
                        Segun=scores[Orden[i]]
                        Terminado='SI'

        if (Segun<0.05 and ValorMax>0.4):
            probCandidates[k]=Groups[cookies[Orden[0]]]

    return probCandidates

#########################################
# THIS FUNCTION MERGES THE DICTIONARIES #
# FOR THE SEMI SUPERVISED LEARNING      #
#########################################

def createOtherDevicesDict(dict1,dict2,dict3):

    OtherDevices=defaultdict(set)
    for k,v in dict1.iteritems():
        for cookie in v:
            OtherDevices[cookie].add(k)
    for k,v in dict2.iteritems():
        for cookie in v:
            OtherDevices[cookie].add(k)
    for k,v in dict3.iteritems():
        for cookie in v:
            OtherDevices[cookie].add(k)

    return OtherDevices



######################################################
# THIS FUNCTION SAVE THE FINAL PREDICTIONS IN A FILE #
######################################################

def writeSolution(file,selected,DeviceList,CookieList):

    header=list()
    header.append('device_id')
    header.append('cookie_id')

    with open(file, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(header)


        for k,v in selected.iteritems():
            row=list()
            items=list()
            row.append(DeviceList[np.int(k)])
            for elem in (v):
                items.append(CookieList[np.int(elem)])
            if len(v)==0:
                items.append('id_10')            
            row.append(' '.join(items))
            spamwriter.writerow(row)


###############################################################################################
# THIS FUNCTION MAKES THE POST PROCESSING ON A TEST                                           #                       
# IT LOOKS FOR DEVICES WHOSE BEST CANDIDATE SCORES LESS THAN 0.05,                            #
# CREATES A NEW SET OF CANDIDATES CONTAINING EVERY COOKIE THAT SHARES AN IP ADDRESS WITH HIM, #
# SCORES THEM WITH XGBOOST AND SELECT THE CANDIDATES FOR THE SUBMISSION                       #
###############################################################################################

def PostAnalysisTest(validatTST,thTST,classifiers,DevicesTest,Cookies,DeviceIPS,CookieIPS,IPDev,IPCoo,Groups,WhosDevice,DevProperties,DictHandle):

    itn=0
    for k,v in validatTST.iteritems():
        itn=itn+1
        if thTST[k]<0.05:

            (fcandK,fcandU)=fullCandidates(k,DevicesTest,Cookies,IPDev,IPCoo,DeviceIPS,DictHandle)

            validatTHK=dict()
            thTHK=dict()
            if(len(fcandK[k])>0):

                (XTHK,OriginalIndexTHK)=createDataSet(fcandK,DevicesTest,Cookies,DeviceIPS,CookieIPS,IPDev,IPCoo,Groups,WhosDevice,DevProperties)

                (tmxK,dmxK)=XTHK.shape
                estimK=np.zeros(tmxK)

                for (classifier,traindev,testdev) in classifiers:
                    estimK=estimK+predictXGBoost(XTHK,classifier)

                estimK=estimK/np.float_(len(classifiers))

                (validatTHK,thTHK)=bestSelection(estimK, OriginalIndexTHK, np.array([1.0,0.90]),Groups)

            validatTHU=dict()
            thTHU=dict()
            if(len(fcandU[k])>0):
                (XTHU,OriginalIndexTHU)=createDataSet(fcandU,DevicesTest,Cookies,DeviceIPS,CookieIPS,IPDev,IPCoo,Groups,WhosDevice,DevProperties)
                (tmxU,dmxU)=XTHU.shape
                estimU=np.zeros(tmxU)

                for (classifier,traindev,testdev) in classifiers:
                    estimU=estimU+predictXGBoost(XTHU,classifier)

                estimU=estimU/np.float_(len(classifiers))

                (validatTHU,thTHU)=bestSelection(estimU, OriginalIndexTHU, np.array([1.0,0.90]),Groups)

              
            if len(validatTHK)>0:
                if len(validatTHU)>0:
                    if(thTHU[k]>(thTHK[k]+0.7)):
                        validatTST[k]=validatTHU[k]
                        thTST[k]=thTHU[k]
                    else:
                        if thTST[k]<=0.025:
                            validatTST[k]=validatTHK[k]
                            thTST[k]=thTHK[k]
                        else:
                            if thTST[k]+0.3<thTHK[k]:
                                validatTST[k]=validatTHK[k]
                                thTST[k]=thTHK[k]
                else:
                    if thTST[k]<=0.025:
                        validatTST[k]=validatTHK[k]
                        thTST[k]=thTHK[k]
                    else:
                        if thTST[k]+0.3<thTHK[k]:
                            validatTST[k]=validatTHK[k]
                            thTST[k]=thTHK[k]
            else:
                validatTST[k]=validatTHU[k]
                thTST[k]=thTHU[k]


    return(validatTST,thTST)


##############################################################################################
# THIS FUNCTION USES THE CLASSIFIER TO MAKE THE PRECICTIONS IN EVERY DEVICE/CANDIDATE COOKIE #
##############################################################################################
    
def Predict(XTST,classifiers):

    (tamTST,dTST)=XTST.shape
    resultadosTST=np.zeros(tamTST)

    for (bst,traindev,testdev) in classifiers:

        pTST=predictXGBoost(XTST,bst)
        resultadosTST=resultadosTST+pTST


    resultadosTST=resultadosTST/np.float_(len(classifiers))
    return resultadosTST

#################################
# THIS FUNCTION LOADS THE MODEL #
#################################

def loadModel(modelpath):

    modelfile=modelpath+os.path.sep+'model.pkl'

    f = open(modelfile, "r")

    nclassifier = pickle.load(f)
    DictOtherDevices = pickle.load(f)

    f.close()

    classifiers=list()

    for i in range(nclassifier):
        classifier = xgb.Booster({'nthread':12})
        classifier.load_model(modelpath+os.path.sep+str(i)+'.model')
        classifiers.append((classifier,set(),set()))
    
    return (classifiers,DictOtherDevices)


#################################
# THIS FUNCTION SAVES THE MODEL #
#################################

def saveModel(modelpath,classifiers,DictOtherDevices):

    d = os.path.dirname(modelpath)

    if not os.path.exists(d):
        os.makedirs(d)

    modelfile=modelpath+os.path.sep+'model.pkl'

    f = open(modelfile, "w")

    pickle.dump(len(classifiers), f)
    
    nclassifier=0

    for (classifier,indtr,indtst) in classifiers:
        classifier.save_model(modelpath+os.path.sep+str(nclassifier)+'.model')
        nclassifier = nclassifier +1
    
    pickle.dump(DictOtherDevices, f)
    f.close()

