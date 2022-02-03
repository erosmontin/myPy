#Eros Montin eros.montin@gmail.com
import os
import json
import uuid
import glob
import requests
import tarfile

import tarfile
from os import listdir
from os.path import isfile, join
import mimetypes
from PIL import Image
import integv


def unTarGz(fname):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()

def unTar(fname):
    tar = tarfile.open(fname, "r:")
    tar.extractall()
    tar.close()



def splitext_(path):
    if len(path.split('.')) > 1:
        # return path.split('.')[0],'.'.join(path.split('.')[-2:])
        S = path.split('.')
        PT = S[0]
        E = ''
        for s in range(1, len(S)):
            if s == 1:
                E = E + S[s]
            else:
                E = E + '.' + S[s]
        return PT, E
    return path, None

def is_json(myjson):
  try:
      json_object = json.loads(myjson)
  except ValueError as e:
      return False
  return True

def getHeadersForRequests():
    return {"Content-Type": "application/json","User-Agent": 'My User Agent 1.0','From': 'theweblogin@iam.com'}
    

def downloadFileFromWeb(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        print("i am downloading " + url + " i'll save it in" + file_name)
        response = requests.get(url,headers=getHeadersForRequests())
        print("done downloading " + url + " i'll save it in" + file_name)
        
        # write to file
        print("writing to file" + file_name)
        file.write(response.content)



class Pathable:
    """extract info from a file position"""

#position '/media/montie01/FOTO/LIFESTORAGE/20211228_000305.jpg'
#path '/media/montie01/FOTO/LIFESTORAGE/'
# filename '/media/montie01/FOTO/LIFESTORAGE/20211228_000305'
# basename '20211228_000305.jpg'
    def __init__(self, position):
        self.position = position
        # self.basename=self.getBaseName(self.position)
    def isDir(self):
        return os.path.isdir(self.position)
    
    def isFile(self):
        return os.path.isfile(self.position)

    def exists(self):
        if self.getExtension() is None:
            return self.isDir()
        return self.isFile()

    def getPosition(self):
        return self.position

    def getPath(self):
        file_name, extension = splitext_(self.position)
        return os.path.dirname(file_name)

    def getBaseName(self):
        return os.path.basename(self.position)
    
    def getBaseNameWithoutExtension(self):
        
        l=splitext_(os.path.basename(self.position))
        return l[0]

    def getExtension(self):
        file_name, extension = splitext_(self.position)
        return extension

    def getFilename(self):
        file_name, extension = splitext_(self.position)

        return file_name

    def getAboluteFilePath(self):
        return os.path.abspath(os.path.expanduser(os.path.expandvars(self.position)))

    def getAbsolutePath(self):
        return self.getPath(os.path.abspath(os.path.expanduser(os.path.expandvars(self.position))))

    def getFullfileNameWIthPrefix(self, prefix):
        N = self.getBaseName()
        P = self.getPath()
        return os.path.join(P, prefix + N)

    def changePath(self,path):
        basename=self.getBaseName()
        self.position=os.path.join(path,basename)
        return self.getPosition()

    def changeBasename(self,name):
        pt=self.getPath()
        self.position=os.path.join(pt,name)
        return self.getPosition()
    def changeFilename(self,bn):
        O=bn +'.' + self.getExtension()
        return self.changeBasename(O)
    
    def addSuffix(self,suf):
        self.position=self.getPostionNameWithSuffix(suf)
        return self.position

    def addPrefix(self,pre):
        self.position=self.getPostionNameWithPrefix(pre)
        return self.position


    def getFullfileNameWIthSuffix(self, suffix,rename=False):
        P = self.getPath()
        N = self.getBaseName()
        E = self.getExtension()
        if E is not None:
            N = N.replace('.' + E, '')
            if rename:
                self.position=os.path.join(P, N + suffix + '.' + E)
            return os.path.join(P, N + suffix + '.' + E)
            
        else:
            if rename:
                self.position=os.path.join(P, N + suffix)
            return os.path.join(P, N + suffix)
    
    def reNameFile(self,newName='notset',E='notset'):
        if E =='notset':
            E = self.getExtension()
        
        if newName =='notset':
            newName = self.getFilename()

        PT=self.getPath()
        if E is None:
            return os.path.join(PT,newName)
        return os.path.join(PT,newName +'.'+ E)
    
    

    def getRandomPostionName(self):
        return self.reNameFile(str(uuid.uuid4()))

    def getNewPositionName(self,newName):
        return self.reNameFile(newName)
    
    # def getNewPositionName(self,newName):
    #     return self.reNameFile(newName)
    
    def getNewPositionExtension(self,ext):
        return self.reNameFile(E=ext)
    
    def getStringForSearchingingInDirectoryallFilesWithExtension(self):
        return self.getNewPositionName('*')
        

    def readJsonFile(self):
        try:
            with open(self.position) as f:
                data = json.load(f)
            return data, True
        except:
            return None,False
    def writeTojsonFile(self,data={"uno":1,"due":2}):
        try:
            with open(self.position, 'w') as outfile:
                json.dump(data, outfile)
                return True
        except:
            return False
    
    def getRandomPostionNameWithSuffix(self,suffix):
        F=self.getRandomPostionName()
        Y=Pathable(F)
        return Y.getFullfileNameWIthSuffix(suffix) 
    
    def getRandomPostionNameWithPrefix(self,prefix):
        F=self.getRandomPostionName()
        Y=Pathable(F)
        return Y.getFullfileNameWIthPrefix(prefix)

    def getPostionNameWithSuffix(self,suffix):
        return self.getFullfileNameWIthSuffix(suffix) 
    
    def getPostionNameWithPrefix(self,prefix):
        return self.getFullfileNameWIthPrefix(prefix)
    
    def getFilesInPositionByExtension(self,ext=None,sort=True):
        if ext is None:
            A = self.getStringForSearchingingInDirectoryallFilesWithExtension()
        else:
            A = self.reNameFile('*',E=ext)

        if sort:
            return sorted(glob.glob(A))
        else:
            return glob.glob(A)
    
    def printFilesInPositionByExtension(self,ext=None,sort=True):

        for t in self.getFilesInPositionByExtension(ext=ext,sort=sort):
            print (t)
    def getListOfFilesInPositionByExtension(self,ext=None,sort=True):
        return [t for t in self.getFilesInPositionByExtension(ext=ext,sort=sort)]

    def getListOfFileInPosition(self):
        if not self.isDir():
            return [self.getPath() +'/' +f for f in listdir(self.getPath()) if isfile(join(self.getPath(), f))]
        else:
            return [self.getPath() +'/' +f for f in listdir(self.getPath())]

    def createPathablePath(self):
        directory=self.getPath()
        if not os.path.exists(directory):
            os.makedirs(directory)
    def getMiMEFileType(self):
        return  mimetypes.guess_type(self.getPosition())
    
    def getFileType(self):
        mime=self.getMiMEFileType()
        return mime[0].split('/')[0]
    
    def isItAnImage(self):
        return self.getFileType()=='image'

    def isItAVideo(self):
        return self.getFileType()=='video'

    def isItCorrupted(self):
        if self.isItAnImage():
            try:
                img = Image.open(self.position) # open the image file
                img.verify() # verify that it is, in fact an image
                return True

            except (IOError, SyntaxError) as e:
                print('Bad file:', self.position) #
                return False
        if self.isItAVideo():
            return integv.verify(self.position,self.getMiMEFileType())



    

    
    
        


import time
import numpy as np
class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
    def getStops(self):
        """Return the stops time."""
        return np.array(self.times).tolist()



