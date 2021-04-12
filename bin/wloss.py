'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import pylab as lab
import numpy as np

e4sp10 = '''
0.0036391059
0.0027566177
0.0022036934
0.0016890782
0.0014077100
0.0013371104
0.0013137705
0.0012763637
0.0012772882
0.0012470149
0.0012547742
0.0012290019
0.0012536801
0.0012190547
0.0012356105
0.0012039845
0.0012398766
0.0011948684
0.0012276612
0.0011988725
0.0012185090
0.0011910938
0.0012176742
0.0011874147
0.0000037839
0.0000043292
0.0000055248
0.0000088139
0.0000088613
0.0000087215
0.0000083858
0.0000086850
0.0000082304
0.0000085543
0.0000081415
0.0000085136
0.0000080652
0.0000084370
0.0000080300
0.0000083910
0.0000078793
0.0000084580
0.0000078763
0.0000084323
0.0000078879
0.0000084464
0.0000078070
0.0000083322
'''.strip().split('\n')
e4sp10 = [float(x) for x in e4sp10]

e4sp100 = '''
0.0041342126
0.0032899405
0.0028100850
0.0023703240
0.0021537163
0.0020720707
0.0020803541
0.0020129522
0.0020378199
0.0019787473
0.0020063701
0.0019535224
0.0019874305
0.0019400995
0.0019709258
0.0019344556
0.0019605996
0.0019257728
0.0019504323
0.0019199757
0.0019465317
0.0019067739
0.0019374229
0.0019107300
0.0000037839
0.0000043319
0.0000040800
0.0000047557
0.0000043002
0.0000045442
0.0000042669
0.0000045063
0.0000041642
0.0000044625
0.0000041347
0.0000044045
0.0000041525
0.0000043952
0.0000040911
0.0000043754
0.0000040614
0.0000043877
0.0000040460
0.0000043634
0.0000040271
0.0000043607
0.0000040277
0.0000043618
'''.strip().split('\n')
e4sp100 = [float(x) for x in e4sp100]

e4sp1000 = '''
0.0076393560
0.0070893178
0.0065348092
0.0061882096
0.0058746384
0.0057583395
0.0057124672
0.0055895736
0.0056152921
0.0055034156
0.0055474667
0.0054373890
0.0054975427
0.0053928373
0.0054663047
0.0053553176
0.0054343333
0.0053340923
0.0054106792
0.0053134430
0.0053876857
0.0052938326
0.0053712521
0.0052744350
0.0000037839
0.0000040119
0.0000034938
0.0000037261
0.0000032557
0.0000034779
0.0000031688
0.0000033739
0.0000031197
0.0000033282
0.0000030796
0.0000032873
0.0000030513
0.0000032600
0.0000030371
0.0000032393
0.0000030232
0.0000032221
0.0000030158
0.0000032086
0.0000029982
0.0000031994
0.0000029854
0.0000031861
'''.strip().split('\n')
e4sp1000 = [float(x) for x in e4sp1000]

e4sp10000 = '''
0.0415419191
0.0421947204
0.0389097631
0.0383935124
0.0357835069
0.0360195301
0.0348518454
0.0348662920
0.0342205353
0.0342173353
0.0337389559
0.0338838845
0.0334355235
0.0334891863
0.0331814699
0.0332894698
0.0330348015
0.0330667868
0.0329218283
0.0327909365
0.0327955484
0.0326654278
0.0326693170
0.0324998945
0.0000037839
0.0000038983
0.0000035705
0.0000035536
0.0000032867
0.0000033315
0.0000032008
0.0000032224
0.0000031421
0.0000031611
0.0000030982
0.0000031306
0.0000030699
0.0000030928
0.0000030461
0.0000030747
0.0000030322
0.0000030536
0.0000030223
0.0000030269
0.0000030109
0.0000030146
0.0000029991
0.0000029994
'''.strip().split('\n')
e4sp10000 = [float(x) for x in e4sp10000]

e16sp10 = '''
0.0039052234
0.0029901555
0.0023816261
0.0018054427
0.0014332832
0.0011515826
0.0010306753
0.0009069968
0.0008443558
0.0007794432
0.0007378339
0.0007083450
0.0006717134
0.0006463836
0.0006259707
0.0006096318
0.0005882080
0.0005732858
0.0005642699
0.0005578736
0.0005407719
0.0005255699
0.0005241250
0.0005187762
0.0000037839
0.0000043995
0.0000053359
0.0000085904
0.0000102349
0.0000133207
0.0000124513
0.0000152416
0.0000135581
0.0000155761
0.0000136177
0.0000155928
0.0000134064
0.0000151713
0.0000133489
0.0000148755
0.0000130966
0.0000144799
0.0000128614
0.0000143153
0.0000126423
0.0000139770
0.0000124078
0.0000135361
'''.strip().split('\n')
e16sp10 = [float(x) for x in e16sp10]

iter = np.arange(24)
e4sp10 = np.array(e4sp10).reshape(2, 24).astype(np.double)
e4sp100 = np.array(e4sp100).reshape(2, 24).astype(np.double)
e4sp1000 = np.array(e4sp1000).reshape(2, 24).astype(np.double)
e4sp10000 = np.array(e4sp10000).reshape(2, 24).astype(np.double)
e16sp10 = np.array(e16sp10).reshape(2, 24).astype(np.double)
kernel = (np.ones(5)/5).astype(np.double)

fig = lab.figure(figsize=(20,4))
#fig = lab.figure(figsize=(24,4))
A = 0.6

tmp = e4sp10
ax1 = lab.subplot(1, 4, 1)
#ax1.plot(tmp[0,:], color='yellowgreen', alpha=0.3)
#x = np.convolve(tmp[0,:], kernel, mode='same');
#x[-1] = tmp[0,-1]; x[-2] = tmp[0,-2]; x[0] = tmp[0,0]; x[1] = tmp[0,1]
ax1.plot(tmp[0,:], color='yellowgreen', marker='.', alpha=A)
ax1.legend(['Total'], loc='lower right', bbox_to_anchor=[0.5, 0.5, 0.5, 0.5]) #'center right')
ax1.xaxis.grid('on', alpha=0.5)
lab.ylabel('Value of Total Loss')
lab.xlabel('Iteration')
ax2 = ax1.twinx()
ax2.plot(tmp[1,:], color='aqua', alpha=0.3)
#x = np.convolve(tmp[1,:], kernel, mode='same');
#x[-1] = tmp[1,-1]; x[-2] = tmp[1,-2]; x[0] = tmp[1,0]; x[1] = tmp[1,1]
ax2.plot(tmp[1,:], color='aqua', marker='.', alpha=A)
ax2.legend(['QA+'], loc='upper right', bbox_to_anchor=[0.5, 0, 0.5, 0.5]) #'center right')
lab.ylabel('Value of QA+ Loss Term')
lab.title('$\\varepsilon=4/255$, $\\xi=10^1$')

ax1 = lab.subplot(1, 4, 2)
ax1.plot(e4sp100[0,:], color='yellowgreen', marker='.',  alpha=A)
ax1.legend(['Total'], loc='upper right') #'center right')
ax1.xaxis.grid('on', alpha=0.5)
lab.ylabel('Value of Total Loss')
lab.xlabel('Iteration')
ax2 = ax1.twinx()
ax2.plot(e4sp100[1,:], color='aqua', marker='.',  alpha=A)
ax2.legend(['QA+'], loc='upper right', bbox_to_anchor=[0, 0, 1, 0.9]) #'center right')
lab.ylabel('Value of QA+ Loss Term')
lab.title('$\\varepsilon=4/255$, $\\xi=10^2$')

#ax1 = lab.subplot(1, 5, 3)
#ax1.plot(e4sp1000[0,:], color='yellowgreen', alpha=A)
#ax1.legend(['Total'], loc='upper right') #'center right')
#lab.ylabel('Value of Total Loss')
#lab.xlabel('Iteration')
#ax2 = ax1.twinx()
#ax2.plot(e4sp1000[1,:], color='aqua', alpha=A)
#ax2.legend(['QA+'], loc='upper right', bbox_to_anchor=[0, 0, 1, 0.9]) #'center right')
#lab.ylabel('Value of QA+ Loss Term')
#lab.title('$\\varepsilon=4/255$, $\\xi=10^3$')

ax1 = lab.subplot(1, 4, 3)
ax1.plot(e4sp10000[0,:], color='yellowgreen', marker='.',  alpha=A)
ax1.legend(['Total'], loc='upper right') #'center right')
ax1.xaxis.grid('on', alpha=0.5)
lab.ylabel('Value of Total Loss')
lab.xlabel('Iteration')
ax2 = ax1.twinx()
ax2.plot(e4sp10000[1,:], color='aqua', marker='.',  alpha=A)
ax2.legend(['QA+'], loc='upper right', bbox_to_anchor=[0, 0, 1, 0.9]) #'center right')
lab.ylabel('Value of QA+ Loss Term')
lab.title('$\\varepsilon=4/255$, $\\xi=10^4$')

ax1 = lab.subplot(1, 4, 4)
ax1.plot(e16sp10[0,:], color='yellowgreen', marker='.',  alpha=A)
ax1.legend(['Total'], loc='lower right', bbox_to_anchor=[0.5, 0.5, 0.5, 0.5]) #'center right')
ax1.xaxis.grid('on', alpha=0.5)
lab.ylabel('Value of Total Loss')
lab.xlabel('Iteration')
ax2 = ax1.twinx()
ax2.plot(e16sp10[1,:], color='aqua', marker='.',  alpha=A)
ax2.legend(['QA+'], loc='upper right', bbox_to_anchor=[0.5, 0, 0.5, 0.5]) #'center right')
lab.ylabel('Value of QA+ Loss Term')
lab.title('$\\varepsilon=16/255$, $\\xi=10^1$')

fig.tight_layout()
#lab.show()
fig.savefig('wloss.svg')
