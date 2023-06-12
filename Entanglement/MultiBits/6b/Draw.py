import matplotlib.pyplot as plt
from Entanglement.MultiBits.DrawHelper import loadTRecoredRs
from Entanglement.MultiBits.DrawHelper import genIterImg

rs_dir = './data_comp'
savePath = './plots_comp/average continue.png'

#* load results
res = loadTRecoredRs(rs_dir)

#* draw & save
img = genIterImg(res)
fig = plt.figure()
plt.imshow(img, origin='lower', aspect='auto', cmap='viridis')
fig.savefig(savePath)