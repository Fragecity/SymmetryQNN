import os
import pickle
import fnmatch
import numpy as np
from collections import namedtuple
from datetime import datetime
from operator import itemgetter

Result = namedtuple('Result', ['x_lst', 'y_lst', 'x_opt', 'accuracy'])

def loadTRecoredRs(rs_dir):

    # Get list of files with their modification dates
    files = [(f, datetime.strptime(f, 'result %m-%d %H-%M.pkl')) for f in os.listdir(rs_dir) if fnmatch.fnmatch(f, 'result*.pkl')]

    # Sort the files based on the datetime
    files.sort(key=itemgetter(1))

    res = []

    # load files in a loop in ascending datetime order
    for file, _ in files:
        with open(os.path.join(rs_dir, file), 'rb') as f:
            resi = pickle.load(f)
            res.append(resi)

    return res

def genIterImg(res):
    res1 = res[0]
    layers = [result[1] for result in res1] # result here is not the tuple defined above
    iterations = [result[0] for result in res1] # results here contains: [num_layers, num_iters, Result_dr, Result_sg, winner]

    img = np.zeros([max(iterations), max(layers)])

    #* plot relative acc between after and before adding sg (sg-dr), output size should be 5*14
    for i, rs in enumerate(res):
        for result in rs:
            img[result[0]-1][result[1]-1] += result[3].accuracy - result[2].accuracy # at least here rs[0] is num_layers

    return img * 100