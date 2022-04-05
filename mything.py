import datetime
import json
class Log():
    """ A Log Class.
    Just append to the log and we take care of the timing.
    You can set the format of the time.
    
    .. note::
        This function is not suitable for sending spam e-mails.

    .. todo:: 
        - Validate all post fields
        - dededede
    """    
  
    def __init__(self,init=None):
        self.format="%d/%m/%Y, %H:%M:%S"
        self.now=datetime.datetime.now()
        self.version='v0.0v'
        self.dflts='procedure'
        self.dflte='ERROR'
        if init is None:
            init="Log (" + self.version + ")"

        self.log=[{"when":self.getFormattedDatetime(self.now),"what":init,"type":"start","settings":{"author":"Eros Montin","mail":"eros.montin@gmail.com","motto":"Forty-six and two are just ahead of me"}}]
    def setTimeFormat(self,f):
        self.format=f
        #  should validate this at some point TODO
        return True

    def getFormattedDatetime(self,t):
        return t.strftime(self.format)
    def getNow(self):
        return self.getFormattedDatetime(datetime.datetime.now())
    def setDefaultType(self,f):
        if isinstance(f,str):
            self.dflts=f
            return True
        else:
            return False
    def getDefaultType(self):
        return self.dflts
    
    def setDefaultError(self,f):
        if isinstance(f,str):
            self.dflte=f
            return True
        else:
            return False
    def getDefaultError(self):
        return self.dflte

    def appendError(self,m=None):
        if m is None:
            m="ERROR"
        self.append(m,self.getDefaultError())

    def append(self,message,type=None,settings=None):
        """append the current message to the log using the time of the call

        Args:
            - message (_type_): The message to be logged.
            - type (_type_, optional): a tag good for automatic identification of type, for example ERROR or DONE. Defaults to "flow". but you can customize to set custom message
            - settings (_type_, optional): a dictoinary of options. Defaults to None.
        """        
        if type is None:
            type=self.getDefaultType()

        self.log.append({"when":self.getNow(),"what":message,"type":type,"settings":settings})
    
    def getWhatHappened(self):
        self.printWhatHappened()

    def printWhatHappened(self):
        """print the events logged
        """        
        for l in self.log:
            print(l)
    
    def getLog(self):
        """gett the events logged
        """
        return self.log
    
    def writeLogAs(self,fn):
        try:
            with open(fn, 'w') as fout:
                json.dump(self.getLog(), fout)
            return True
        except:
            return False
    
    def saveLogAs(self,fn):
        return self.writeLogAs(fn)
    



import subprocess

class BashIt(object):
    def __init__(self) -> None:
        self.bashCommand=None
        self.Log=Log('Bash')
    def setCommand(self,comm):
        self.bashCommand=comm
        self.Log.append(f'added command {comm}')
    def getCommand(self):
        return self.bashCommand
    def run(self):
        self.Log.append(f'running')
        bashCommand=self.getCommand()
        if bashCommand is not None:
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            self.Log.append(f'running {bashCommand}')
            self.output, self.error = process.communicate()
            self.Log.append(f'completed {bashCommand}')
            return True
        else:
            return False


    def getBashError(self):
        return self.error
    
    def getBashOutput(self):
        return self.output