'''
Copyright (C) 2020-2021 Mo Zhou <cdluminate@gmail.com>
Released under the Apache-2.0 License.
'''
import pylab as lab
import numpy as np

e4sp10 = '''
1.0971641541
0.2661794126
0.1439510286
0.1717298031
0.1141558811
0.1089500040
0.0769365728
0.0924363211
0.0818059146
0.0972240865
0.0717044026
0.0644484162
0.0641587004
0.0599376485
0.0583688356
0.0742397085
0.0669365376
0.0654452443
0.0540778190
0.0747289658
0.0564763397
0.0725676939
0.0565276705
0.0533721335
0.0001836773
0.0028125744
0.0021640451
0.0011045494
0.0007661097
0.0007233915
0.0005333290
0.0006904893
0.0005385295
0.0004467997
0.0003991046
0.0003910359
0.0004738964
0.0003659719
0.0003416982
0.0003726038
0.0002230042
0.0003333597
0.0002793144
0.0003390081
0.0002874604
0.0002093112
0.0002992359
0.0002385821
'''.strip().split('\n')
e4sp10 = [float(x) for x in e4sp10]

e4sp100 = '''
1.1097608805
0.2900490165
0.1937693357
0.2434587777
0.1208473220
0.1298015565
0.0945863426
0.0990360901
0.1022857875
0.0814502239
0.1017439887
0.0759416074
0.0832295269
0.0725379661
0.0758037418
0.0775843859
0.0788925588
0.0748400539
0.0648240075
0.0698487014
0.0829102844
0.0649779588
0.0729146749
0.0693076029
0.0001836772
0.0003603179
0.0005174096
0.0005580405
0.0002460789
0.0001728197
0.0001956670
0.0001632818
0.0002065986
0.0001368822
0.0001333432
0.0001489756
0.0001335576
0.0001268162
0.0001328162
0.0000954870
0.0001176265
0.0000899211
0.0001091256
0.0000830503
0.0000885701
0.0000848146
0.0000858609
0.0000924893
'''.strip().split('\n')
e4sp100 = [float(x) for x in e4sp100]

e4sp1000 = '''
0
'''.strip().split('\n')
e4sp1000 = [float(x) for x in e4sp1000]

e4sp10000 = '''
3.2015120983
5.3256793022
2.1438415051
1.4172098637
1.0447499752
1.0520360470
0.7702566385
0.7444332242
0.7743114829
0.6504150033
0.6277613044
0.5826765299
0.6092577577
0.5214490891
0.5701878071
0.5168851614
0.5282954574
0.4649601579
0.5427594185
0.4842484593
0.4982892275
0.5163050890
0.4544746876
0.4674662948
0.0001961842
0.0004955024
0.0001765590
0.0001165576
0.0000788809
0.0000876197
0.0000560066
0.0000589945
0.0000611104
0.0000502329
0.0000442917
0.0000443056
0.0000435668
0.0000387638
0.0000410946
0.0000394730
0.0000375692
0.0000349051
0.0000404161
0.0000362099
0.0000361374
0.0000392076
0.0000322145
0.0000343553
'''.strip().split('\n')
e4sp10000 = [float(x) for x in e4sp10000]

e16sp10 = '''
1.0972491503
0.3293713331
0.1273372471
0.1249925718
0.0955415368
0.0806757510
0.0710522756
0.0623753965
0.0592008792
0.0600071400
0.0458578356
0.0591376834
0.0611218698
0.0526743196
0.0474101566
0.0375180393
0.0461780988
0.0322707444
0.0459960438
0.0398767851
0.0422808528
0.0322830454
0.0371450782
0.0407247506
0.0002004503
0.0017186706
0.0019251619
0.0018059879
0.0013114694
0.0009764315
0.0006999001
0.0009041607
0.0004906078
0.0005302362
0.0004089773
0.0004569484
0.0004345492
0.0002798470
0.0003191177
0.0002097576
0.0003537225
0.0002282123
0.0001788458
0.0001680787
0.0002167259
0.0002105917
0.0001682081
0.0001546303
'''.strip().split('\n')
e16sp10 = [float(x) for x in e16sp10]

iter = np.arange(24)
e4sp10 = np.array(e4sp10).reshape(2, 24).astype(np.double)
e4sp100 = np.array(e4sp100).reshape(2, 24).astype(np.double)
#e4sp1000 = np.array(e4sp1000).reshape(2, 24).astype(np.double)
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
fig.savefig('wloss-sop.svg')
