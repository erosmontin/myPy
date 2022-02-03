import im


REF='/data/PROJECTS/HIPSEGENTATION/hip_segmentation/input/p05-1.nii.gz'
R=im.Imaginable(inputFileName=REF)

IM='/data/test/test/roi2/segmentationp05-1Left.nii.gz'
A=im.Imaginable(inputFileName=IM)

A.reshapeOverImage(R)


IM2='/data/test/test/roi2/segmentationp05-1Right.nii.gz'
B=im.Imaginable(inputFileName=IM2)

A.addImage(B)

A.writeImageAs('/data/o.nii.gz')