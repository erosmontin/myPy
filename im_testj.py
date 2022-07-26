import im
THEIMAGE='/home/eros/Desktop/DCM/roi/rois/segmentationmriLeft.nii'
THEIMAGE2='/home/eros/test.nii.gz'


A=im.Imaginable(THEIMAGE) 

A.reshapeImageToNewGrid(newSpacing=(1,1,1))
A.writeImageAs(THEIMAGE2) 