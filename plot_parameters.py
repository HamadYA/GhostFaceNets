import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
fontP = FontProperties()

db="calfw"

if (db=="agedb"):
    agedb = [98.100,97.32,97.6,96.4,94.4,97.05,96.983,96.633,97.05,95.85,95.617,96.07,93.22, 96.9167, 98, 96.18, 97.58]
    accuracies=agedb
    flops = [1000,577.5,900,1100,900,626.1,626.1,451.7,451.7,161.9,161.9,439.8,66.9, 60.3,215.7, 60.3,215.7]
    nets=["VarGFaceNet", "ShuffleFaceNet 1.5", "MobileFaceNet", "MobileFaceNetV1", "ProxylessFaceNAS", "MixFaceNet-M","ShuffleMixFaceNet-M","MixFaceNet-S","ShuffleMixFaceNet-S","MixFaceNet-XS","ShuffleMixFaceNet-XS","MobileFaceNets","ShuffleFaceNet 0.5", "GhostFaceNet-2 (MS1MV3) (ours)", "GhostFaceNet-1 (MS1MV3) (ours)", "GhostFaceNet-2 (MS1MV2) (ours)", "GhostFaceNet-1 (MS1MV2) (ours)"]
    marker=['X', '2', '<', 'h', 'h', 'h', '1', 'x', '+', 'D', '3', 'v', 'v','o','o','o', 'o']
    # marker=['o', '.', 'H', 'x', '+', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v','1','3','2', '4', '5']

    save_path = "./agedb.png"
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(accuracies, flops, 'o')
    plt.ylabel("Accuracy (%) ",fontsize=16)
    plt.xlabel("MFLOPs",fontsize=16)
    plt.ylim([93, 98.3])
    plt.xlim([30, 1200])

elif(db=="calfw"):
    accuracies = [94.47, 95.15,
                    95.2, 92.55, 95.05,
                    95.48, 95.5, 95.67, 95.63,
                    95.60, 95.93, 95.53, 95.88]
    flops = [1100, 1022,
                933.3, 900, 577.5,
                0.925, 0.99, 1.68, 1.75]
    nets = ["MobileFaceNetV1", "VarGFaceNet",
            "MobileFaceNet", "ProxylessFaceNAS", "ShuffleFaceNet 1.5x",
            "PocketNetS-128", "PocketNetS-256", "PocketNetM-128", "PocketNetM-256", 
            "GhostFaceNet-2 (MS1MV3) (ours)", "GhostFaceNet-1 (MS1MV3) (ours)", "GhostFaceNet-2 (MS1MV2) (ours)", "GhostFaceNet-1 (MS1MV2) (ours)"
            ]

    marker = ['2', '<',
                '1', 'x', '+',
                '*',
                'h', 'h', 'h', 'h',
                'o', 'o', 'o', 'o']
    save_path = "./calfw.png"
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(accuracies, flops, 'o')
    plt.ylabel("Accuracy (%) ",fontsize=26)
    plt.xlabel("MFLOPs",fontsize=26)
    plt.ylim([86, 97])
    plt.xlim([30, 1200])

elif(db=="cplfw"):
    accuracies = [87.17, 88.55,
                    89.22, 84.17, 88.50,
                    89.68,
                    89.63, 88.93, 90, 90.03]
    params = [3.4, 5.0,
                2.0, 3.2, 2.6,
                1.35,
                0.925, 0.99, 1.68, 1.75]
    nets = ["MobileFaceNetV1", "VarGFaceNet",
            "MobileFaceNet", "ProxylessFaceNAS", "ShuffleFaceNet 1.5x",
            "Distill-DSE-LSE",
            "PocketNetS-128 (ours)", "PocketNetS-256 (ours)", "PocketNetM-128 (ours)", "PocketNetM-256 (ours)"]

    marker = ['2', '<',
                '1', 'x', '+',
                '*',
                'o', 'o', 'o', 'o']
    save_path = "./cplfw.pdf"
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(accuracies, params, 'o')
    plt.ylabel("Accuracy (%) ",fontsize=26)
    plt.xlabel("Params",fontsize=26)
    plt.ylim([80, 91])
    plt.xlim([0.8, 6])


elif(db=="cfp"):
    accuracies = [97.56, 95.8, 98.5,
                    96.9, 94.7, 96.9,
                    94.19, 92.59,
                    94.21, 93.34, 95.07, 95.56]
    params = [4.5, 3.4, 5.0,
                2.0, 3.2, 2.6,
                1.35, 0.5,
                0.925, 0.99, 1.68, 1.75]
    nets = ["ShuffleFaceNet 2x", "MobileFaceNetV1", "VarGFaceNet",
            "MobileFaceNet", "ProxylessFaceNAS", "ShuffleFaceNet 1.5x",
            "Distill-DSE-LSE", "ShuffleFaceNet 0.5x",
            "PocketNetS-128 (ours)", "PocketNetS-256 (ours)", "PocketNetM-128 (ours)", "PocketNetM-256 (ours)"]

    marker = ["X", '2', '<',
                '1', 'x', '+',
                '*', '3',
                'o', 'o', 'o', 'o']
    save_path = "./cfp.pdf"
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(accuracies, params, 'o')
    plt.ylabel("Accuracy (%) ",fontsize=26)
    plt.xlabel("Params",fontsize=26)
    plt.ylim([80, 98])
    plt.xlim([0.8, 6])
    
elif(db=="lfw"):
    accuracies = [99.85,99.67,99.7,99.4,99.2,99.683,99.6,99.65,99.583,99.6,99.533,99.55,99.23,99.3, 99.6833,99.7333, 99.65, 99.7667]
    flops = [1000,577.5,900,1100,900,626.1,626.1,451.7,451.7,161.9,161.9,439.8,66.9,1000, 60.3,215.7, 60.3,215.7]
    nets=["VarGFaceNet", "ShuffleFaceNet 1.5", "MobileFaceNet", "MobileFaceNetV1", "ProxylessFaceNAS", "MixFaceNet-M","ShuffleMixFaceNet-M","MixFaceNet-S","ShuffleMixFaceNet-S","MixFaceNet-XS","ShuffleMixFaceNet-XS","MobileFaceNets","ShuffleFaceNet 0.5","AirFace",  "GhostFaceNet-2 (MS1MV3) (ours)", "GhostFaceNet-1 (MS1MV3) (ours)", "GhostFaceNet-2 (MS1MV2) (ours)", "GhostFaceNet-1 (MS1MV2) (ours)"]
    # marker=['o', '.', 'H', 'x', '+', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v','1','3','2', '4', '1']
    marker=['X', '2', '<', 'h', 'h', 'h', '1', 'x', '+', 'D', '3', 'v', 'v','v','o','o', 'o', 'o']
    save_path = "./lfw.png"
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(accuracies, flops, 'o')
    plt.ylabel("Accuracy (%) ",fontsize=16)
    plt.xlabel("MFLOPs",fontsize=16)
    plt.ylim([99.15, 99.87])
    plt.xlim([30, 1200])

# elif(db=="megaface"):
#     accuracies = [93.9,93.0,95.2,91.3,82.8,94.26,94.24,92.23,93.60,89.40,89.24,90.16,96.52]
#     flops = [1000,577.5,900,1100,900,626.1,626.1,451.7,451.7,161.9,161.9,439.8,1000, 60.3,215.7, 60.3,215.7]
#     nets=["VarGFaceNet", "ShuffleFaceNet 1.5", "MobileFaceNet", "MobileFaceNetV1", "ProxylessFaceNAS", "MixFaceNet-M","ShuffleMixFaceNet-M","MixFaceNet-S","ShuffleMixFaceNet-S","MixFaceNet-XS","ShuffleMixFaceNet-XS","MobileFaceNets","AirFace", "GhostFaceNet-2 (MS1MV3) (ours)", "GhostFaceNet-1 (MS1MV3) (ours)", "GhostFaceNet-2 (MS1MV2)", "GhostFaceNet-1 (MS1MV2)"]
#     marker=['o', '.', 'H', 'x', '+', 'v', 'v', 'v', 'v', 'v', 'v','1','3']
#     save_path = "./megaface.png"
#     plt.figure()
#     fig, ax = plt.subplots()
#     plt.plot(accuracies, flops, 'o')
#     plt.ylabel("TAR at FAR1e–6 ",fontsize=16)
#     plt.xlabel("MFLOPs",fontsize=16)
#     plt.ylim([82.1, 96.7])
#     plt.xlim([50, 1200])

# elif(db=="megafacer"):
#     accuracies = [95.6,94.6,96.8,93.0 ,84.8 ,95.83 ,95.22  ,93.79, 95.19, 91.04, 91.03, 92.59, 97.93]
#     flops = [1000,577.5,900,1100,900,626.1,626.1,451.7,451.7,161.9,161.9,439.8,1000, 60.3,215.7, 60.3,215.7]
#     nets=["VarGFaceNet", "ShuffleFaceNet 1.5", "MobileFaceNet", "MobileFaceNetV1", "ProxylessFaceNAS", "MixFaceNet-M","ShuffleMixFaceNet-M","MixFaceNet-S","ShuffleMixFaceNet-S","MixFaceNet-XS","ShuffleMixFaceNet-XS","MobileFaceNets","AirFace", "GhostFaceNet-2 (MS1MV3) (ours)", "GhostFaceNet-1 (MS1MV3) (ours)", "GhostFaceNet-2 (MS1MV2)", "GhostFaceNet-1 (MS1MV2)"]
#     marker=['o', '.', 'H', 'x', '+', 'v', 'v', 'v', 'v', 'v', 'v','1','3']
#     save_path = "./megafacer.png"
#     plt.figure()
#     fig, ax = plt.subplots()
#     plt.plot(accuracies, flops, 'o')
#     plt.ylabel("TAR at FAR1e–6 ",fontsize=16)
#     plt.xlabel("MFLOPs",fontsize=16)
#     plt.ylim([84.1, 98.1])
#     plt.xlim([30, 1200])

elif(db=="IJB-B"):
    accuracies = [92.9,92.3,92.8,92.0,87.1,91.55,91.47,90.17,90.94,88.48,87.86,91.2463,93.1159, 90.5258, 92.191]

    flops = [1000,577.5,900,1100,900,626.1,626.1,451.7,451.7,161.9,161.9,60.3,215.7, 60.3,215.7]
    nets=["VarGFaceNet", "ShuffleFaceNet 1.5", "MobileFaceNet", "MobileFaceNetV1", "ProxylessFaceNAS", "MixFaceNet-M","ShuffleMixFaceNet-M","MixFaceNet-S","ShuffleMixFaceNet-S","MixFaceNet-XS","ShuffleMixFaceNet-XS", "GhostFaceNet-2 (MS1MV3) (ours)", "GhostFaceNet-1 (MS1MV3) (ours)", "GhostFaceNet-2 (MS1MV2) (ours)", "GhostFaceNet-1 (MS1MV2) (ours)"]
    # marker=['o', '.', 'H', 'x', '+', 'v', 'v', 'v', 'v', 'v', 'v','1','3','2','4']
    marker=['X', '2', '<', 'h', 'h', 'h', '1', 'x', '+', 'v', 'v','o','o','o', 'o']
    save_path = "./ijbb.png"
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(accuracies, flops, 'o')
    plt.ylabel("TAR at FAR1e–4 ",fontsize=16)
    plt.xlabel("MFLOPs",fontsize=16)
    plt.ylim([87, 93.5])
    plt.xlim([30, 1200])

elif(db=="IJB-C"):
    accuracies = [94.7,94.3,94.7,93.9,89.7,93.42,93.5,92.30,93.08,90.73,90.43,93.45,94.94, 92.6574, 94.058]

    flops = [1000,577.5,900,1100,900,626.1,626.1,451.7,451.7,161.9,161.9,60.3,215.7, 60.3,215.7]
    nets=["VarGFaceNet", "ShuffleFaceNet 1.5", "MobileFaceNet", "MobileFaceNetV1", "ProxylessFaceNAS", "MixFaceNet-M","ShuffleMixFaceNet-M","MixFaceNet-S","ShuffleMixFaceNet-S","MixFaceNet-XS","ShuffleMixFaceNet-XS", "GhostFaceNet-2 (MS1MV3) (ours)", "GhostFaceNet-1 (MS1MV3) (ours)", "GhostFaceNet-2 (MS1MV2) (ours)", "GhostFaceNet-1 (MS1MV2) (ours)"]
    # marker=['o', '.', 'H', 'x', '+', 'v', 'v', 'v', 'v', 'v', 'v','1','3','2','4']
    marker=['X', '2', '<', 'h', 'h', 'h', '1', 'x', '+', 'v', 'v','o','o','o', 'o']
    save_path = "./ijbc.png"
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(accuracies, flops, 'o')
    plt.ylabel("TAR at FAR1e–4 ",fontsize=16)
    plt.xlabel("MFLOPs",fontsize=16)
    plt.ylim([88, 95.2])
    plt.xlim([30, 1200])

p=[]
for i in range(len(accuracies)):
    if "ours" in nets[i]:
        plt.plot(flops[i], accuracies[i], marker[i],markersize=12,markeredgecolor='red',label=nets[i])
    else:
        plt.plot(flops[i], accuracies[i], marker[i],markersize=12,label=nets[i])

plt.grid()

plt.legend(numpoints=1, loc='lower right',fontsize=8,ncol=2)
plt.savefig(save_path, format='png', dpi=600)
plt.close()
