
import sys
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from DP_Misra_Gries.Testing import Test, TestKSweep
from collections import Counter

def make_skewed_stream(n, k, a):
    n_keys = k
    
    raw = np.random.pareto(a, size=n_keys) + 1
    weights = raw / raw.sum()
    keys = np.random.randint(0, 2**32, size=n_keys, dtype=np.uint32)

    stream = np.random.choice(keys, size=n, p=weights)
    return stream

    
def run_experiments(a):
    stream = make_skewed_stream(1000000, 10000, a)
    counts = Counter(stream)

    print("stream length:", len(stream))
    print("distinct observed elements:", len(counts))
    print("starting Experiment")


    epsilons = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    ks = [500, 1000, 2000]
    delta = 1/(len(stream)*len(stream))
    trials = 10
    experiment_name = "PowerLaw" + str(a)
    for k in ks:
        ####### [ Graphs to expect out of this: 4]
        # Regular AE 
        # Top-K MAE 
        # Recall@k 
        # Max Error 
        #######
        Test(stream, k, epsilons, delta, trials=trials, ExperimentName=experiment_name)
        return


    ks = [100, 250, 500, 1000, 2000, 4000, 6000, 8000, 10000]
    epsilons=[0.05, 0.1, 0.5, 1.0]
    for epsilon in epsilons:
        ####### [ Graphs to expect out of this: 4]
        # Regular AE 
        # Top-K MAE 
        # Recall@k 
        # Max Error 
        #######
        TestKSweep(stream, ks, epsilon, delta, trials=trials, ExperimentName=experiment_name)

np.random.seed(42)
run_experiments(1.0)

np.random.seed(42)
run_experiments(4.0)
