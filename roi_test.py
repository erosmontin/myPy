import im

T='/data/PROJECTS/HIPSEGENTATION/hip_segmentation/input/ResizedWithZoom/numcv_4p03-1Left.nii.gz_segmentation.nii'
# T='/data/PROJECTS/HIPSEGENTATION/hip_segmentation/output/p03roi.nii.gz'
R='/data/PROJECTS/HIPSEGENTATION/hip_segmentation/output/p03-1roi.nii.gz'


A=im.ROIable()
r=im.Imaginable(inputFileName=R)
t=im.Imaginable(inputFileName=T)

r.reshapeOverImage(t)
A.setReference(r)

A.setTest(t)
A.setReferenceThreshold(1)
A.setTestThreshold(1)
print(A.getJaccard())
print(A.getDice())
print(A.getFalseNegativeError())
print(A.getFalsePostiveError())
print(A.getHahusdorf())
A.testImages('/data/')
