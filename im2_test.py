import im

IM='/data/PROJECTS/HIPSEGENTATION/hip_segmentation/input/ResizedWithZoom/numcv_4p03-1Left.nii.gz'


A=im.Imaginable(inputFileName=IM)

A.writeImage(outputFileName="/data/T.nii.gz")

