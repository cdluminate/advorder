'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import numpy as np
import pylab as lab

DATA='''
Rand & 0.233 & 0.291 & 0.309 & 0.318 & 0.320
Beta & 0.249 & 0.313 & 0.360 & 0.368 & 0.382
PSO  & 0.280 & 0.341 & 0.381 & 0.382 & 0.385
NES  & 0.309 & 0.380 & 0.416 & 0.431 & 0.438
SPSA & 0.292 & 0.365 & 0.407 & 0.421 & 0.433
'''.strip()

data = [ [x[2], x[4], x[6], x[8], x[10]] for x in (y.split() for y in DATA.split('\n'))]
data = [ list(map(float, x)) for x in data]
data = np.array(data)
print(data)

lab.figure()
X = [100,500,1000,5000,10000]
lab.plot(X, data[0,:], color='coral', marker='.')
lab.plot(X, data[1,:], color='gold', marker='^')
lab.plot(X, data[2,:], color='yellowgreen', marker='*')
lab.plot(X, data[3,:], color='aqua', marker='h')
lab.plot(X, data[4,:], color='cornflowerblue', marker='D')
#lab.axis([0, 10000, 0.2, 0.5 ])
lab.axis('auto')
lab.grid('on', linestyle='dotted')
lab.xlabel('Query Budget $Q$')
lab.ylabel('$\\tau_N$')
lab.legend(['Rand', 'Beta', 'PSO', 'NES', 'SPSA'], loc='lower right', fancybox=True)
#lab.xscale('log')
#lab.show()
lab.savefig('qbudcurve.svg')
