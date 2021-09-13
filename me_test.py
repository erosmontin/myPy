# import tmpme as me
import me

clock=me.Timer()
clock.start()
thestring='/data/tmp/bb/another/thefile.nii.gz.cc'
A=me.Pathable(thestring)

L=A.getFullfileNameWIthPrefix('test__')
print(L)

print(A.getStringForSearchingingInDirectoryallFilesWithExtension())

print(A.getNewPositionExtension('altro'))

L=A.getFullfileNameWIthSuffix('__test')
print(L)

print(A.getRandomPostionName())
thestring='/data/tmp/bb/another/thefile.nii.gz'
A=me.Pathable(thestring)

L=A.getFullfileNameWIthPrefix('test__')
print(L)

L=A.getFullfileNameWIthSuffix('__test')
print(L)

thestring='/data/tmp/bb/another/thefile.nii'
A=me.Pathable(thestring)

L=A.getFullfileNameWIthPrefix('test__')
print(L)

L=A.getFullfileNameWIthSuffix('__test')
print(L)


thestring='/data/tmp/bb/another/thefile'
A=me.Pathable(thestring)

L=A.getFullfileNameWIthPrefix('test__')
print(L)

L=A.getFullfileNameWIthSuffix('__test')
print(L)


thestring='/data/PERSONALPROJECTS/myPy/test.py'
A=me.Pathable(thestring)
L=A.exists()
print('Does '+ thestring + " exist " + str(L))


thestring='/data/PERSONALPROJECTS/myPy/testddd.py'
A=me.Pathable(thestring)
L=A.exists()
print('Does '+ thestring + " exist " + str(L))



thestring='/data/PERSONALPROJECTS/myPy'
A=me.Pathable(thestring)
L=A.exists()
print('Does '+ thestring + " exist " + str(L))

print(clock.stop())
thestring='/data/PERSONALPROJECTS/myPy/a'
A=me.Pathable(thestring)
L=A.isDir()
print('Does '+ thestring + " exist " + str(L))


thestring='/data/tmp/ac.json'
A=me.Pathable(thestring)
di='pppppppppp'
A.writeTojsonFile(di)
L=A.exists()
print('Does '+ thestring + " exist " + str(L))

print(type(di))

print(clock.stop())

print(A.getRandomPostionNameWithPrefix('prova__'))
print(A.getRandomPostionNameWithSuffix('__prova'))

print(A.getPostionNameWithPrefix('prova__'))
print(A.getPostionNameWithSuffix('__prova'))


print(A.getPostionNameWithPrefix('prova/'))


print(clock.stop())

A.printFilesInPositionByExtension()

A.printFilesInPositionByExtension('nii')


print(clock.avg())
print(clock.cumsum())
print(clock.getStops())