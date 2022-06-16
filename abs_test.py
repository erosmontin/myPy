import im


REF='/data/MYDATA/Poirot_RIccardo_Example/SNR_array.nii.gz'
R=im.Imaginable(inputFileName=REF)
R.applyABS()

IM='/data/MYDATA/Poirot_RIccardo_Example/SNR_perf.nii'
A=im.Imaginable(inputFileName=IM)
A.applyABS()

IMf='/data/MYDATA/Poirot_RIccardo_Example/USNR_3D.nii.gz'

B=im.Imaginable(inputFileName=IMf)
B.applyABS()


    
A.writeImageAs('/data/MYDATA/Poirot_RIccardo_Example/P.nii')
R.writeImageAs('/data/MYDATA/Poirot_RIccardo_Example/S.nii')
B.writeImageAs('/data/MYDATA/Poirot_RIccardo_Example/U.nii')