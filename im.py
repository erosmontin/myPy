# this needs the me environment
import os 
import SimpleITK as sitk
from SimpleITK.SimpleITK import ThresholdSegmentationLevelSetImageFilter
import numpy as np
import pylab
import nibabel as nib

class IndexTracker(object):
        def __init__(self, ax, X):
            self.ax = ax

            self.X = X
            ax.set_title('use scroll wheel to navigate the image size(' + str(X.shape) +')')
            rows, cols, self.slices  = X.shape
            self.ind = self.slices//2

            self.im = ax.imshow(self.X[:, :,self.ind])
            self.update()

        def onscroll(self, event):
            # print("%s %s" % (event.button, event.step))
            if event.button == 'up':
                self.ind = (self.ind + 1) % self.slices
            else:
                self.ind = (self.ind - 1) % self.slices
            self.update()

        def update(self):
            self.im.set_data(self.X[:,:,self.ind])
            self.ax.set_ylabel('slice ' + str(self.ind) )
            self.im.axes.figure.canvas.draw()
            # print(int(self.ind))
        
        def onclick(self,event):
            # print('you pressed', event.button, event.xdata, event.ydata)
            # print(type(event.button))
           
            if event.button == 2:
               print('you pressed', event.button, event.xdata, event.ydata)
               print('value %s' % self.X[int(np.floor(event.ydata)),int(np.floor(event.xdata)),self.ind])


""" @brief creates a simpleitk random image

    @param imageSize=[20,20,20],imageResolution=[1.0,1.0,1.0],imageOrigin=[0.0,0.0,0.0],imageDirection=[[1.0,0.0,0.0],[0,1.0.0,0.0],[0.0,0.0,1.0]]
"""

def createSITKImagefromNumpyArray(nda, imageResolution=[1.0,1.0,1.0],imageOrigin=[0.0,0.0,0.0],imageDirection=[1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1]):
    img = sitk.GetImageFromArray(adjustNumpyArrayForITK(nda))
    img.SetDirection(imageDirection)
    img.SetOrigin(imageOrigin)
    img.SetSpacing(imageResolution)
    return img

def createRandomSITKImage(imageSize=[20,20,20],imageResolution=[1.0,1.0,1.0],imageOrigin=[0.0,0.0,0.0],imageDirection=[1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1]):
    nda=np.random.random(imageSize)
    return createSITKImagefromNumpyArray(nda,imageResolution,imageOrigin,imageDirection)

def createLabelMapSITKImage(imageSize=[20,20,20],imageResolution=[1.0,1.0,1.0],imageOrigin=[0.0,0.0,0.0],imageDirection=[1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1],values=[0,1]):
    nda = np.random.choice(values, size=imageSize)
    nda=np.uint8(nda)
    return createSITKImagefromNumpyArray(nda,imageResolution,imageOrigin,imageDirection)

""" @brief creates an imaginable random image
    @param imageSize=[20,20,20],imageResolution=[1.0,1.0,1],imageOrigin=[0.0,0.0,0.0],imageDirection=[[1.0.0,0.0,0.0],[0.0,1.0.0,0.0],[0.0,0.0,1.0]]
"""
def createImaginableFormSITKImage(sitk,imageName='randomImaginable.nii.gz'):
    A=Imaginable()
    A.setImage(sitk)
    A.setOutputFileName(imageName)
    return A

def createRandomImaginable(imageName='randomImaginable.nii.gz',imageSize=[20,20,20],imageResolution=[1.0,1.0,1.0],imageOrigin=[0.0,0.0,0.0],imageDirection=[1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1]):
    sitk=createRandomSITKImage(imageSize,imageResolution,imageOrigin,imageDirection)
    return createImaginableFormSITKImage(sitk,imageName)

def createRandomLabelmapImaginable(imageName='randomImaginable.nii.gz',imageSize=[20,20,20],imageResolution=[1.0,1.0,1.0],imageOrigin=[0.0,0.0,0.0],imageDirection=[1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1],values=[0,1]):
    sitk=createLabelMapSITKImage(imageSize,imageResolution,imageOrigin,imageDirection,values)
    return createImaginableFormSITKImage(sitk,imageName)

class Imaginable():
    def __init__(self,**kwargs ):
        #inputFileName=lslsll.nii.gz
        self.InputFileName = None
        self.OutputFileName = None
        self.Image = None
        if kwargs is None:
            print("start")
        elif len(kwargs)==1:
            v=kwargs.values()
            vi=iter(v)
            self.setInputFileName(next(vi))
            
        else:
            if 'inputFileName' in kwargs:
                self.setInputFileName(kwargs['inputFileName'])
            if 'outputFileName' in kwargs:
                self.setOutputFileName(kwargs['outputFileName'])
                
    def getInputFileName(self):
        try:
            return self.InputFileName
        except:
            return None
    def setInputFileName(self,fn):
        self.InputFileName = fn
    def getOutputFileName(self):
        try: 
            return self.OutputFileName
        except:
            return None
    def setOutputFileName(self,fn):
        self.OutputFileName = fn
    def getImage(self):
        if self.Image is None:
            if self.getInputFileName() is None:
                print("No input Image set")
                return None
            self.readImage()
        return self.Image
        
    def getImageSize(self):
        image=self.getImage()
        return image.GetSize()

    def getImageDirections(self):
        image=self.getImage()
        return image.GetDirection()
    
    def getImageSpacing(self):
        image=self.getImage()
        return image.GetSpacing()
    
    def getImageOrigin(self):
        image=self.getImage()
        return image.GetOrigin()
        
    def getImageDimension(self):
        image=self.getImage()
        return image.GetDimension()
    
    def getImageNumberOfComponentsPerPixel(self):
        image=self.getImage()
        return image.GetNumberOfComponentsPerPixel()

    def getImagenumberOfComponents(self):
        image=self.getImage()
        return image.GetNumberOfComponentsPerPixel()

    def getImagePixelType(self):
        image=self.getImage()
        return image.GetPixelIDValue(), image.GetPixelIDTypeAsString()

    def setImage(self,im):
        self.Image = im
    
    def getImageArray(self):
        image=self.getImage()
        return sitk.GetArrayFromImage(image)
        
    def setImageFromNibabel(self,filename):
        img = nib.load(filename)
        data = img.get_fdata()
        self.setImageArray(data)


    def __del__(self):
        print("I'm being automatically destroyed. Goodbye!")

    def setImageArray(self,array):
        #input is a nd array #         nda = sitk.GetArrayFromImage(image) or image.getImageArray()
        image=self.getImage()
        if image is None:
            array=adjustNumpyArrayForITK(array)
            self.setImage(sitk.GetImageFromArray(array, isVector=False))
            print('no Information on Image')
        else:
            nda = sitk.GetImageFromArray(array)
            nda.CopyInformation(image)
            nda.SetSpacing(self.getImageSpacing())
            nda.SetOrigin(self.getImageOrigin())
            nda.SetDirection(self.getImageDirections())
            self.setImage(nda)
    def setImageInformationFromExternalImage(self,externalImage):
        image=self.getImage()
        image.SetSpacing(externalImage.getImageSpacing())
        image.SetOrigin(externalImage.getImageOrigin())
        image.SetDirection(externalImage.getImageDirections())
        self.setImage(image)

    def readImage(self):
        self.setImage(sitk.ReadImage(self.getInputFileName()))
    def writeImageAs(self,filename):
        self.writeImage(outputFileName=filename)
    def writeImage(self,**kwargs):
        if kwargs is not None:
            if 'outputFileName' in kwargs:
                self.setOutputFileName(kwargs['outputFileName'])
        sitk.WriteImage(self.getImage(), self.getOutputFileName())

    def printImageInfo(self):
        image= self.getImage()
        print(image)
        for key in image.GetMetaDataKeys():
            print("\"{0}\":\"{1}\"".format(key, image.GetMetaData(key)))
    def getImageAsNumpyArray(self):
        return sitk.GetArrayFromImage(self.getImage())

    def viewAxial(self):
        fig, ax = pylab.subplots(1, 1)
        tracker = IndexTracker(ax, self.getImageAsNumpyArray())
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        fig.canvas.mpl_connect('button_press_event', tracker.onclick)
        pylab.show()
   
    def resetImage(self):
        filen=self.getInputFileName()
        if filen is not None:
            self.readImage()
            return True
        return False

    def resizeImage(self,newSize):
        image=self.getImage()
        dimension = image.GetDimension()
        reference_physical_size = np.zeros(image.GetDimension())
        reference_physical_size[:] = [round((sz-1)*spc) if sz*spc>mx  else mx for sz,spc,mx in zip(image.GetSize(), image.GetSpacing(), reference_physical_size)]

        reference_origin = image.GetOrigin()
        reference_direction = image.GetDirection()

        
        reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(newSize, reference_physical_size) ]
        centered_transform =sitk.Transform()
        
        self.setImage(regridSitkImage(image, newSize, centered_transform,  sitk.sitkLinear, reference_origin, reference_spacing, reference_direction, 0.0,image.GetPixelIDValue()))
        return True

    def cropImage(self,lower,upper):
        image=self.getImage()
        self.setImage(sitk.Crop(image,lower,upper))
    
    def reshapeOverImage(self,targetImage,**kwargs):
        if kwargs is None:
            print("start")
        else:
            if 'interpolator' in kwargs:
                interpolator= kwargs['interpolator']
            else:
                interpolator =  sitk.sitkLinear
        self.reshapeImageToNewGrid(pixelValue=0,
        intrepolator= interpolator,
        newSize= targetImage.getImageSize(),
        newOrigin=targetImage.getImageOrigin(),
        newDirections=targetImage.getImageDirections(),
        newSpacing=targetImage.getImageSpacing()
        )


        
    def reshapeImageToNewGrid(self,**kwargs):
        image=self.getImage()
        

        if kwargs is None:
            print("start")
        else:
            if 'pixelValue' in kwargs:
                default_value= kwargs['pixelValue']
            else:
                default_value=0
            if 'interpolator' in kwargs:
                interpolator= kwargs['interpolator']
            else:
                interpolator =  sitk.sitkLinear
            if 'transform' in kwargs:
                reansform=kwargs['transform']
            else:
                transform=sitk.Transform()
            if 'newSize' in kwargs:
                newSize=kwargs['newSize']
            else:
                newSize=self.getImageSize()
            if 'newOrigin' in kwargs:
                newOrigin=kwargs['newOrigin']
            else:
                newOrigin=self.getImageOrigin()
            
            if 'newSpacing' in kwargs:
                newSpacing=kwargs['newSpacing']
            else:
                newSpacing=self.getImageSpacing()

            if 'newDirections' in kwargs:
                newDirections=kwargs['newDirections']
            else:
                newDirections=self.getImageDirections()
            

        self.setImage(regridSitkImage(image, newSize, transform , interpolator, newOrigin, newSpacing, newDirections, default_value,image.GetPixelIDValue()))


    
    def translateImage(self,T):
        image=self.getImage()
        dimension=self.getImageDimension()
        translation = sitk.TranslationTransform(dimension, T)
        self.setImage(transformImage(image,translation))
    def getCoordinatesFromIndex(self,P):
        image=self.getImage()
        return image.TransformContinuousIndexToPhysicalPoint(P)

    def getImageCenterCoordinates(self):
        return self.getCoordinatesFromIndex(np.array(self.getImageSize())/2.0)

    def composeImageCosinesTransform(self,transform):
        image=self.getImage()
        dimension=self.getImageDimension()
        cosines = sitk.AffineTransform(dimension)
        cosines.SetCenter(self.getImageOrigin())
        return sitk.CompositeTransform([cosines,transform])

    def translateImageAffine(self,T):
        image=self.getImage()
        dimension=self.getImageDimension()
        
        transform = sitk.AffineTransform(dimension)
        transform.SetTranslation(T)

        transform.SetCenter(self.getImageOrigin())
        thetransform=self.composeImageCosinesTransform(transform)
        self.setImage(transformImage(image,thetransform))
        return thetransform

    def scaleImageAffine(self,S,**kwargs):
        image=self.getImage()
        dimension=self.getImageDimension()
        
        if kwargs is not None:
            if 'transformCenter' in kwargs:
                center=kwargs['transformCenter']
            else :
                center = self.getImageCenterCoordinates()
   
        transform = sitk.AffineTransform(dimension)
        #transform.SetMatrix(self.getImageDirections())
        
        matrix = np.array(transform.GetMatrix()).reshape((dimension,dimension))
        for s in range(dimension):
            matrix[s,s] = S[s]
        transform.SetMatrix(matrix.ravel())
        transform.SetCenter(center)
        thetransform=self.composeImageCosinesTransform(transform)

        self.setImage(transformImage(image,thetransform))
        return thetransform

class ROIable(Imaginable):
    def __init__(self):
        self.Reference=None
        self.Test=None
        self.Overlap=None
        self.ReferenceThreshold=1
        self.TestThreshold=1
    
    def getReference(self):
        return self.Reference
    def getTest(self):
        return self.Test
    def setReference(self,reference):
        self.OverlapFiler=None
        self.Reference =reference
    def setTest(self,test):
        self.OverlapFiler=None
        self.Test=test
    def setOverlapFilter(self,overlap):
        self.Overlap =overlap

    def getOverlapFilter(self):
        ov=self.OverlapFiler
        if ov is None:
            R=self.getReference()
            T=self.getTest()
            t=T.getImage()
            r=R.getImage()
            if (r is not None) & (t is not None):
                ov=sitk.LabelOverlapMeasuresImageFilter()
                thr=self.getReferenceThreshold()
                tht=self.getTestThreshold()
                ov.Execute(r==thr, t==tht)
                self.setOverlapFilter(ov)
                return ov
            else:
                return None

        else:
            return ov
    def getReferenceThreshold(self):
        return self.ReferenceThreshold
    def setReferenceThreshold(self,t):
        self.OverlapFiler=None
        self.ReferenceThreshold=t
    
    def getTestThreshold(self):
        return self.TestThreshold
    def setTestThreshold(self,t):
        self.OverlapFiler=None
        self.TestThreshold=t


    def getJaccard(self):
        ov=self.getOverlapFilter()
        if ov is not None:
            return ov.GetJaccardCoefficient()
        else:
            return None
    
    def getDice(self):
        ov=self.getOverlapFilter()
        if ov is not None:
            return ov.GetDiceCoefficient()
        else:
            return None

    def getSimilarity(self):
        ov=self.getOverlapFilter()
        if ov is not None:
            return ov.GetVolumeSimilarity()
        else:
            return None
    
    def getFalseNegativeError(self):
        ov=self.getOverlapFilter()
        if ov is not None:
            return ov.GetFalseNegativeError()
        else:
            return None
    
    def getFalsePostiveError(self):
        ov=self.getOverlapFilter()
        if ov is not None:
            return ov.GetFalsePositiveError()
        else:
            return None
    
    def getHahusdorf(self):
        ov = sitk.HausdorffDistanceImageFilter()
        R=self.getReference()
        T=self.getTest()
        t=T.getImage()
        r=R.getImage()

        thr=self.getReferenceThreshold()
        tht=self.getTestThreshold()
        ov.Execute(r==thr, t==tht)
        if ov is not None:
            return ov.GetHausdorffDistance()
        else:
            return None
    
    def testImages(self,pt):
        r=os.path.join(pt,'r.nii.gz')
        t=os.path.join(pt,'t.nii.gz')
        R=self.getReference()
        R.writeImage(outputFileName=r)
        T=self.getTest()
        T.writeImage(outputFileName=t)
        


def transformImage(image, transform, **kwargs):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    default_value = 0
    interpolator = sitk.sitkCosineWindowedSinc
    reference_image=image

    if kwargs is None:
        print("start")
    else:
        if 'pixelValue' in kwargs:
            default_value= kwargs['pixelValue']
        if 'interpolator' in kwargs:
            interpolator= kwargs['interpolator']
        if 'referenceImage' in kwargs:
            reference_image=kwargs['referenceImage']
        
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)
def regridSitkImage(image, newSize, transform , interpolator, newOrigin, newSpacing, newDirections, default_value,pixelidvalue):
    return sitk.Resample(image, newSize, transform , interpolator, newOrigin, newSpacing, newDirections, default_value,pixelidvalue)


        
        
def adjustNumpyArrayForITK(array):
    # ITK's Image class does not have a bracket operator. It has a GetPixel which takes an ITK Index object as an argument, which is an array ordered as (x,y,z). This is the convention that SimpleITK's Image class uses for the GetPixel method as well. While in numpy, an array is indexed in the opposite order (z,y,x).

    if(len(array.shape)==3):
        return np.swapaxes(array,0,2)
    elif(len(array.shape)==2):
        return np.swapaxes(array,0,1)
    else:
        print("numpy shap enot yet seen!! in adjustNumpyArrayForITK")
        return None
        