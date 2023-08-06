import numpy as np
import matplotlib.pyplot as plt

def generate_samples_in_circle(num_samples, center=(0.5, 0.5), radius=0.2):
    samples = []
    while len(samples) < num_samples:
        x, y = np.random.uniform(center[0]-radius, center[0]+radius, 2)
        if np.sqrt((x - center[0])**2 + (y - center[1])**2) <= radius:
            samples.append(((x, y), 1))
    return samples


def generate_samples_out_circle(num_samples, center=(0.5, 0.5), radius=0.2):
    samples = []
    while len(samples) < num_samples:
        x, y = np.random.uniform(0, 1, 2)  # Generate a point within the square
        if np.sqrt((x - center[0])**2 + (y - center[1])**2) > radius:  # If the point is outside the circle
            samples.append(((x, y), -1))  # add the point to the samples
    return samples


def genx1x2Samples(num_samples_each):
    type1Set = generate_samples_in_circle(num_samples_each, center=(0.5, 0.5), radius=0.2)
    type2Set = generate_samples_out_circle(num_samples_each, center=(0.5, 0.5), radius=0.2)
    return type1Set, type2Set

#* test and plot generated samples

type1Set, type2Set = genx1x2Samples(150)

#* split data into train, test, and validation sets then save
trainSet_1, testSet_1, valSet_1 = type1Set[:90], type1Set[90:120], type1Set[120:]
trainSet_2, testSet_2, valSet_2 = type2Set[:90], type2Set[90:120], type2Set[120:]

trainSet = trainSet_1 + trainSet_2
testSet = testSet_1 + testSet_2
valSet = valSet_1 + valSet_2

saveRoot = './Data/'
np.save(saveRoot+'trainSetSmall.npy', trainSet)
np.save(saveRoot+'testSetSmall.npy', testSet)
np.save(saveRoot+'valSetSmall.npy', valSet)

def plotSet(dataSet,color,marker,label):
    for data, label in dataSet:
        plt.scatter(data[0], data[1], c=color, marker=marker)

#* plot the data
plotSet(trainSet, 'r', 'o', 'train')
plotSet(testSet, 'b', 'x', 'test')
plotSet(valSet, 'g', '+', 'val')
plt.legend()
plt.savefig(saveRoot+'trainTestValSmall.png')
plt.show()