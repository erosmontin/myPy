import im

TB='/data/PROJECTS/HIPSEGENTATION/hip_segmentation/input/ResizedWithZoom/numcv_4p03-1Left.nii.gz_segmentation.nii'
T='/data/PROJECTS/HIPSEGENTATION/hip_segmentation/input/ResizedWithZoom/p03-1Left.nii.gz'
R='/data/PROJECTS/HIPSEGENTATION/hip_segmentation/output/p03-1roi.nii.gz'


A=im.ROIable()


r=im.Imaginable(inputFileName=R)
#we take the info from the reference
t=im.Imaginable(inputFileName=T)
t.setImageFromNibabel(TB)

#we subdivided the images in left and right but the principal roi is for the full hip
r.reshapeOverImage(t)
A.setReference(r)

A.setTest(t)
A.setReferenceThreshold(1)
A.setTestThreshold(3)
print(A.getJaccard())
print(A.getDice())
print(A.getFalseNegativeError())
print(A.getFalsePostiveError())
print(A.getHahusdorf())
A.testImages('/data/')
