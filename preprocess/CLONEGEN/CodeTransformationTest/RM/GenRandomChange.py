
import os 
import math 
import time
import random
import argparse


parser = argparse.ArgumentParser() 
parser.add_argument('filepath',type=str, help='please input the count file path')
# parser.add_argument('action',type=int,help='actions')
args=parser.parse_args()
filepath = args.filepath    
# action = args.action
 
def get_conut_result(file_dir):
    F=[]
    for root,dirs,files in os.walk(file_dir):
        for filename in files:
            if(os.path.splitext(filename)[1]=='.count'):
                F.append(filename)
    return F
 

def get_file_number(filename):
    count=0
    for i in filename:
        if(str.isnumeric(i)):
            count+=1
    number=filename[:count]
    return int(number)

def gen_random_data(files):
    random.seed(time.time())
    transformable = []
    for filename in files:
        absFile=os.path.join(filepath,filename)
        print(filename)
        with open(absFile,'r') as fileHandle:
            filenumber=get_file_number(filename)
            changedCount=0
            count=int(fileHandle.readline().strip())
            if(filenumber==1):
                changedCount=0
            elif(filenumber==13):
                changedCount=1
                transformable.append(str(filenumber))  
            elif(count>0):
                changedCount=random.randint(1,count)
                transformable.append(str(filenumber))
            noChangeCount=count-changedCount
            changeVariable=['1']*changedCount
            noChangeVariable=['0']*noChangeCount
            variable=changeVariable+noChangeVariable
            # r=random.random
            x=random.randint(1,100000)
            random.seed(x)
            random.shuffle(variable)    
            saveVariable=os.path.join(filepath,os.path.splitext(filename)[0]+'.random')
            with open(saveVariable,'w') as saveFile:
                result='\n'.join(variable)
                saveFile.write(result)
            with open(os.path.join(filepath, "trans_actions.txt"),'w') as f:
                res = '\n'.join(transformable)
                f.write(res)

if __name__ == '__main__':
    files=get_conut_result(filepath)
    gen_random_data(files)


