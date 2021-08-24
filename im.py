# this needs the me environment
from os import extsep
import SimpleITK as sitk
import numpy as np
import pylab
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

class Imaginable():
    def __init__(self,**kwargs ):
        self.InputFileName = None
        self.OutputFileName = None
        self.Image = None
        if kwargs is None:
            print("start")
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
        

    def setImage(self,im):
        self.Image = im
        

    def readImage(self):
        self.setImage(sitk.ReadImage(self.getInputFileName()))

    def writeImage(self):
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
    #https://stackoverflow.com/questions/48065117/simpleitk-resize-images
    def resizeImage_(self,newSize):
        image=self.getImage()
        dimension = image.GetDimension()
        reference_physical_size = np.zeros(image.GetDimension())
        reference_physical_size[:] = [round((sz-1)*spc) if sz*spc>mx  else mx for sz,spc,mx in zip(image.GetSize(), image.GetSpacing(), reference_physical_size)]

        reference_origin = image.GetOrigin()
        reference_direction = image.GetDirection()

        
        reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(newSize, reference_physical_size) ]

        newImage = sitk.Image(newSize, image.GetPixelIDValue())
        newImage.SetOrigin(reference_origin)
        newImage.SetSpacing(reference_spacing)
        newImage.SetDirection(reference_direction)
        reference_center = np.array(newImage.TransformContinuousIndexToPhysicalPoint(np.array(newImage.GetSize())/2.0))
    
        transform = sitk.AffineTransform(dimension)
        transform.SetMatrix(reference_direction)
        # transform.SetFixedParameters(reference_origin)
        transform.SetTranslation(np.array(image.GetOrigin()) - reference_origin)
    
        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0))
        centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        
        centered_transform =sitk.CompositeTransform([transform,centering_transform])
        # self.setImage(sitk.Resample(image, newImage, centered_transform, sitk.sitkLinear, 0.0))
        self.setImage(sitk.Resample(image, newSize, centered_transform,  sitk.sitkLinear, reference_origin, reference_spacing, reference_direction, 0.0,image.GetPixelIDValue()))
        return True
    
    def resizeImage(self,newSize):
        image=self.getImage()
        dimension = image.GetDimension()
        reference_physical_size = np.zeros(image.GetDimension())
        reference_physical_size[:] = [round((sz-1)*spc) if sz*spc>mx  else mx for sz,spc,mx in zip(image.GetSize(), image.GetSpacing(), reference_physical_size)]

        reference_origin = image.GetOrigin()
        reference_direction = image.GetDirection()

        
        reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(newSize, reference_physical_size) ]
        centered_transform =sitk.Transform()
        
        self.setImage(sitk.Resample(image, newSize, centered_transform,  sitk.sitkLinear, reference_origin, reference_spacing, reference_direction, 0.0,image.GetPixelIDValue()))
        return True

    def cropImage(self,lower,upper):
        image=self.getImage()
        self.setImage(sitk.Crop(image,lower,upper))



        
        

    