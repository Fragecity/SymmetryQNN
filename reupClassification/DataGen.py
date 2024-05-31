# %% import packages
import numpy as np
from numpy import pi


# %% Generate data by a condition functions and a data generator
def generate_constraint_samples(num_samples, data_gen, cond_lst):
    samples = []
    while len(samples) < num_samples:
        data = next(data_gen)
        conds = [cond(data) for cond in cond_lst]
        if all(conds):
            samples.append(data)


# %% Generate Reupload-like data
def data_gen_reupload(data_range=(-0.4 * pi, 0.4 * pi), radius=0.7 / 3 * pi):
    center = (0, 0)
    while 1:
        point = np.random.uniform(*data_range, 2)
        label1 = is_incircle(point, center, 2 * radius)
        label2 = is_incircle(point, center, radius)
        yield (point, int(label1) + int(label2))


# %%  Utils
def is_incircle(point, center, radius):
    bias = np.array(point) - np.array(center)
    return np.linalg.norm(bias) <= radius


def is_condition(point):
    x, y = point
    return y >= x


def is_true(point):
    return True


#%% main
if __name__ == "__main__":
    data_gen = data_gen_reupload()
    num_data = 2 ^ 11
    num_bias_data = 2 ^ 11

    data_set = generate_constraint_samples(num_data, data_gen, [is_true])
    bias_data_set = generate_constraint_samples(num_bias_data, data_gen, [is_condition])

    print("Generated data\n")

    saveRoot = './Data/'
    np.save(saveRoot+'data.npy', data_set)
    np.save(saveRoot+'bias_data.npy', bias_data_set)

    print("Saved data")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

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
