# My Python Packages
Utils for an easy coding life from Dr. Eros Montin, PhD.

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
this is my image package based on [*SimpleITK*](https://simpleitk.org/) 

(because we love [*ITK*](https://itk.org/))

[*Dr. Eros Montin, PhD*](http://me.biodimensional.com)

**46&2 just ahead of me!**
