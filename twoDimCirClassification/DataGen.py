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


def generate_samples_in_circle_tl(num_samples, center=(0.5, 0.5), radius=0.2):
    """adds a condition that y>=x"""
    samples = []
    while len(samples) < num_samples:
        x, y = np.random.uniform(center[0]-radius, center[0]+radius, 2)
        if np.sqrt((x - center[0])**2 + (y - center[1])**2) <= radius and y>=x:
            samples.append(((x, y), 1))
    return samples


def generate_samples_out_circle_tl(num_samples, center=(0.5, 0.5), radius=0.2):
    """adds a condition that y>=x"""
    samples = []
    while len(samples) < num_samples:
        x, y = np.random.uniform(0, 1, 2)  # Generate a point within the square
        if np.sqrt((x - center[0])**2 + (y - center[1])**2) > radius and y>=x:  # If the point is outside the circle
            samples.append(((x, y), -1))  # add the point to the samples
    return samples

def generate_samples_in_circle_cond(num_samples, center=(0.5, 0.5), radius=0.2, cond_expr=""):
    samples = []
    while len(samples) < num_samples:
        x, y = np.random.uniform(center[0]-radius, center[0]+radius, 2)

        condition_satisfied = eval(cond_expr) if cond_expr else True
        if np.sqrt((x - center[0])**2 + (y - center[1])**2) <= radius and condition_satisfied:
            samples.append(((x, y), 1))
    return samples


def generate_samples_out_circle_cond(num_samples, center=(0.5, 0.5), radius=0.2, cond_expr=""):
    samples = []
    while len(samples) < num_samples:
        x, y = np.random.uniform(0, 1, 2)  # Generate a point within the square

        condition_satisfied = eval(cond_expr) if cond_expr else True
        if np.sqrt((x - center[0])**2 + (y - center[1])**2) > radius and condition_satisfied:  # If the point is outside the circle
            samples.append(((x, y), -1))  # add the point to the samples
    return samples


def genx1x2Samples(num_samples_each):
    type1Set = generate_samples_in_circle(num_samples_each, center=(0.5, 0.5), radius=0.2)
    type2Set = generate_samples_out_circle(num_samples_each, center=(0.5, 0.5), radius=0.2)
    return type1Set, type2Set

def genx1x2Samples_tl(num_samples_each):
    type1Set = generate_samples_in_circle_tl(num_samples_each, center=(0.5, 0.5), radius=0.2)
    type2Set = generate_samples_out_circle_tl(num_samples_each, center=(0.5, 0.5), radius=0.2)
    return type1Set, type2Set

def genx1x2Samples_cond(num_samples_each, cond_expr=""):
    type1Set = generate_samples_in_circle_cond(num_samples_each, center=(0.5, 0.5), radius=0.2, cond_expr=cond_expr)
    type2Set = generate_samples_out_circle_cond(num_samples_each, center=(0.5, 0.5), radius=0.2, cond_expr=cond_expr)
    return type1Set, type2Set

#* test and plot generated samples

type1Set, type2Set = genx1x2Samples_cond(60, cond_expr="")  # y>=x and x<=0.5
postFix = "small60"

#* split data into train, test, and validation sets then save
trainSet_1, testSet_1 = type1Set[:36], type1Set[36:48]
trainSet_2, testSet_2 = type2Set[:36], type2Set[36:48]

trainSet = trainSet_1 + trainSet_2
testSet = testSet_1 + testSet_2

saveRoot = './Data/'
np.save(saveRoot+'trainSet'+postFix+'.npy', trainSet)
np.save(saveRoot+'testSet'+postFix+'.npy', testSet)

def plotSet(dataSet,color,marker,label):
    for data, label in dataSet:
        plt.scatter(data[0], data[1], c=color, marker=marker)

#* plot the circle in the plot first
circle1 = plt.Circle((0.5, 0.5), 0.2, color='k', fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle1)

#* plot the data
plotSet(trainSet, 'r', 'o', 'train')
plotSet(testSet, 'b', 'x', 'test')
plt.legend()
plt.savefig(saveRoot+'trainTest'+postFix+'.png')
plt.show()
print("gen done")