import im
import me
import numpy as np


#i want to subdivide in two images half left andf half right


# R.resizeImageInVoxelSpace([0,0,0],[160,320,120])

# R.writeImageAs('/data/t1.nii.gz')

            
# R.resetImage()

# R.resizeImageInVoxelSpace([0,80,0],[160,220,120],spacing='calculate')

# R.writeImageAs('/data/t2.nii.gz')

# SS=[0.34, 0.34, 0.9]
# OL=R.getCoordinatesFromIndex(outputImageOriginStartRegion)
# FL='/data/t2.nii.gz'
# R.resetImage()
# R.reshapeImageToNewGrid(newSize=[320, 320, 120],
#     newOrigin=OL,
#     newSpacing=SS)
# R.writeImageAs(FL)



# S.scaleImageAffine([0.5,0.5,0.5])

# print(R.isPointInside([-2.0,0.0,10]))
# print(R.isPointInside([-20000.0,0.0,10]))



REF='/data/PROJECTS/HIPSEGENTATION/hip_segmentation/input/p05-1.nii.gz'
ROI='/data/PROJECTS/HIPSEGENTATION/hip_segmentation/output/p05-1roi.nii.gz'


transformCenterPoint=[80,160,60]
outputImageSize=[320,320,120]
resolution=[0.34, 0.34, 0.9]
outputImageOriginStartRegion=[0,100,0]

angles=range(-30,30,2)
translations=range(-10,10,2)
scaling=np.linspace(0.70,1.5,10)

radius = range(0,3)
noiselevels =np.linspace(0,20,21)

L=me.Pathable(REF)

roipt=f'/data/test/ROI/x/{L.getBaseNameWithoutExtension()}'
imagept=f'/data/test/ROI/y/{L.getBaseNameWithoutExtension()}'









                #erode dilate
                #noise
                #sharp
                #lowpass

