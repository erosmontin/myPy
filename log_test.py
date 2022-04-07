import mything

L=mything.Log('prova')


L.append('uno')
L.append('udue')
L.append('utre')

L.writeLogAs('/data/t.json')


L2=mything.Log('new start')
L2.appendFullLog(L)
L2.append("EEEEEEEEEEEEEEEEE")
L2.writeLogAs('/data/t22.json')