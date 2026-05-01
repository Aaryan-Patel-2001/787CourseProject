
import sys
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from DP_Misra_Gries.Testing import Test, TestKSweep
from collections import Counter

def make_skewed_stream(n, k, a):
    raw = np.random.power(a, n)
    result = np.minimum((raw * k).astype(int), k)
    return result

def run_experiments(a):
    stream = make_skewed_stream(1000000, 10000, a)
    counts = Counter(stream)

    print("stream length:", len(stream))
    print("distinct observed elements:", len(counts))
    print("starting Experiment")

    k = 100
    epsilons = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    delta = 1e-6 
    trials = 3
    experiment_name = "PowerLaw" + str(a)
    Test(stream, k, epsilons, delta, trials=trials, ExperimentName=experiment_name)

    ks = [5, 10, 50, 100, 150, 200, 500]
    epsilon=1.0
    TestKSweep(stream, ks, epsilon, delta, trials=trials, ExperimentName=experiment_name)

run_experiments(5.0)
run_experiments(10.0)
