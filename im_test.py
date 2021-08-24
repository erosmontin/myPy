import im

IM='/data/PROJECTS/HIPSEGENTATION/hip_segmentation/input/p03.nii.gz'

A=im.Imaginable(inputFileName=IM)

A.cropImage([0,0,0],[160,0,0])
A.resizeImage([320,320,120])
A.setOutputFileName('a.nii.gz')
A.writeImage()
