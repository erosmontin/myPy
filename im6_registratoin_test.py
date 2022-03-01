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



R0='/data/test/EM_MASK.nii'
R1='/data/test/UISNR_MASK.nii.gz'

 



a=im.RegistrationUX(R1,R0)
l=a.register2()

