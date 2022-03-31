import im

IM='/data/test.mha'


A=im.Imaginable(inputFileName=IM)

# A.writeImage(outputFileName="/data/T.nii.gz")

L=A.savePointsCloudAs('/data/PC.dat')



