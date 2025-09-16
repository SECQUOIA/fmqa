import fmqa
import numpy as np
import dimod
import matplotlib.pyplot as plt

def two_complement(x, scaling=True):
    '''
    Evaluation function for binary array
    of two's complement representation.

    example (when scaling=False):
    [0,0,0,1] => 1
    [0,0,1,0] => 2
    [0,1,0,0] => 4
    [1,0,0,0] => -8
    [1,1,1,1] => -1
    '''
    
    # val, n = 0, len(x)
    # print(type(x[0]))
    # for i in range(n):
    #     val += (1<<(n-i-1)) * x[i] * (1 if (i>0) else -1)
    # return val * (2**(1-n) if scaling else 1)
    
    # x = [int(v) for v in x]
    val, n = 0, len(x)
    for i in range(n):
        v = int(x[i])
        val += (1 << (n - i - 1)) * v * (1 if (i > 0) else -1)
    return val * (2**(1 - n) if scaling else 1)

# This is an evaluator of two's complement representation,
# while its output is scaled to [-1,1].
# We fix the input length to 16,
# and generate initial dataset of size 5 for training.

xs = np.random.randint(2, size=(5,16))
ys = np.array([two_complement(x) for x in xs])

# Based on the dataset, train a FMBQM model.
model = fmqa.FMBQM.from_data(xs, ys)

# We use simulated annealing from dimod package 
# here to solve the trained model.
sa_sampler = dimod.samplers.SimulatedAnnealingSampler()

# We repeat taking 3 samples at once and updating 
# the model for 15 times (45 samples taken in total).
for _ in range(15):
    res = sa_sampler.sample(model, num_reads=3)
    xs = np.r_[xs, res.record['sample']]
    ys = np.r_[ys, [two_complement(x) for x in res.record['sample']]]
    model.train(xs, ys)
    

plt.plot(ys, 'o')
plt.xlabel('Selection ID')
plt.ylabel('value (scaled)')
plt.ylim([-1.0,1.0])
plt.show()