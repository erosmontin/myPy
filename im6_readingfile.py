import im
import me
import numpy as np
import SimpleITK as sitk
#open a 5D file and see if i can loop thorough the dimensions and get the values of the data as i expect them to be
#IM=cat(4,ones(10,20,30)+j*onexit()es(10,20,30),2*ones(10,20,30)+j*2*ones(10,20,30),3*ones(10,20,30)+3j*ones(10,20,30))
#F=cat(5,IM,IM+1,IM+1)
#save_nii(make_nii(F,[5 5 5]),'/data/test/test.nii.gz')
#size(10,20,30,3,3)



a=im.Imaginable()



# R0='/data/test/EM_H.nii.gz'
R0='/data/test/test.nii.gz'

C=im.Poirot()
C.setHField(H)
C.setEField(E)
C.setSigmaFromUXSpace(S)
#a=C.getCovarianceMatrix()
#print(a)
A=C.getEHSNR()
C.getLog()
#A.writeImageAs('/data/test/p.nii.gz')