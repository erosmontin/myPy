# My Python Packages
Utils for an easy coding life from Dr. [*Eros Montin, PhD*](http://me.biodimensional.com).

### me
don't need the environement me to be activated
### Pathable
Everything you need to work with path

```
import me
A=me.Pathable('/data/tmp/rrr.nii.gz')
print(A.getFullfileNameWIthPrefix('first_elab_'))

```

```
$> /data/tmp/first_elab_rrr.nii.gz

```

### Timer
A Timer class to measure the time

```
clock =me.timer()

#optional
clock.start()

print(clock.stop())


print(clock.avg())
print(clock.cumsum())
print(clock.getStops())
```


## im
this is my image package based on [*SimpleITK*](https://simpleitk.org/) 2.0
(because we love [*ITK*](https://itk.org/) version 5.0)
```
from me import im

IM='/data/PROJECTS/HIPSEGENTATION/hip_segmentation/input/p03.nii.gz'

A=im.Imaginable(inputFileName=IM)
B=im.Imaginable(inputFileName=IM)

A.cropImage([0,0,0],[160,0,0])
A.setOutputFileName('pre.nii.gz')
A.resizeImage([100,100,100])
A.translateImage([5,5,5])
A.writeImage(outputFileName="T.nii.gz")


B.cropImage([0,0,0],[160,0,0])
B.resizeImage([100,100,100])
B.translateImageAffine([5,5,5])
B.writeImage(outputFileName="TA.nii.gz")

```

## References
If you enjoyed this package consider citing one of my article in your papers. Here is the link to my [*publications*](http://me.biodimensional.com)

**46&2 just ahead of me!**
