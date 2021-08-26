import im

IM='/data/PROJECTS/HIPSEGENTATION/hip_segmentation/input/p03.nii.gz'

A=im.Imaginable(inputFileName=IM)
B=im.Imaginable(inputFileName=IM)

A.cropImage([0,0,0],[160,0,0])
A.setOutputFileName('pre.nii.gz')
A.resizeImage([100,100,100])
A.translateImage([5,5,5])
A.writeImage(outputFileName="T.nii.gz")


B.cropImage([0,0,0],[160,0,0])
B.resizeImage([100,100,100])
B.translateImageAffine([5,5,5])
B.writeImage(outputFileName="TA.nii.gz")

P=B.getImageCenterCoordinates()
B.scaleImageAffine([1.0,1.0,1.0/0.4],transformCenter=B.getCoordinatesFromIndex([75,40,40]))
S=B.getImageSize()
B.cropImage([0,0,0],[0,0,round(S[2]*0.4)])
B.writeImage(outputFileName="TAS.nii.gz")

A.reshapeImageToNewGrid(newSize=[320, 320, 120],
newOrigin=B.getCoordinatesFromIndex([20, 20, 0]),
newSpacing=[0.4, 0.4, 1])
A.writeImage(outputFileName='Zoom.nii.gz')



