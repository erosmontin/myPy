import im, me
import numpy as np
    
test='/home/montie01/Desktop/482f41fb-3c63-3108-88ed-0413880cc7abT.nii'

OUT='/data/TEST1/TESSTestDataHead'


D=me.Pathable(test)


for t in D.getFilesInPositionByExtension():
    A=im.Imaginable(fileName=t)
    # A.cropImage([0,0,0],[0,0,400])
    A.resizeImage(newSize=[100,100,100])
    T=me.Pathable(t)
    T.changeExtension('nii')
    T.changePath(OUT)
    T.ensureDirectoryExistence()
    A.writeImageAs(T.getPosition())

