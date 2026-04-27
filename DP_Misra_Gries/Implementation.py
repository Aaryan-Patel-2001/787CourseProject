import numpy as np 
from diffprivlib.mechanisms import Laplace # type: ignore
import math 
import heapq

class DP_Misra_Gries: 

    def __init__(self, stream, k, epsilon, delta):
        if k <= 0:
            raise ValueError("k must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")
        self.stream = stream 
        self.k = k 
        self.epsilon = epsilon 
        self.delta = delta 

        
    def misra_gries(self): 
        counters = {("__dummy__", i): 0 for i in range(self.k)}

        for x in self.stream: 

            if x in counters: 
                counters[x] += 1 

            elif all(count >= 1 for count in counters.values()):
                # Decrement all counters
                for key in list(counters.keys()): 
                    counters[key] -= 1 

            else:
                # Replace the smallest zero-count key according to deterministic order.
                zero_keys = [key for key, count in counters.items() if count == 0]
                key_to_remove = min(zero_keys, key=str)

                del counters[key_to_remove]
                counters[x] = 1

        # Remove dummy counters before privatization/output
        counters = {
            key: count
            for key, count in counters.items()
            if not (isinstance(key, tuple) and key[0] == "__dummy__")
        }

        return counters
    
    def addNoiseToCounters(self, counters):
        LaplaceNoiseGen = DiffprivlibLaplaceNoise(self.epsilon)

        shared_noise = LaplaceNoiseGen.sample()

        noisy = {}
        for key, count in counters.items():
            independent_noise = LaplaceNoiseGen.sample()
            noisy[key] = count + shared_noise + independent_noise

        return noisy
    
    def threshold_noisy_counts(self, noisy):
        threshold = 1.0 + (2.0 * math.log(3.0 / self.delta)) / self.epsilon

        return {
            key: value
            for key, value in noisy.items()
            if value >= threshold
        }
    
    def compute(self): 
        counters = self.misra_gries()
        noisy = self.addNoiseToCounters(counters)
        threshold = self.threshold_noisy_counts(noisy)
        return threshold
 

class DiffprivlibLaplaceNoise:
    def __init__(self, epsilon: float, sensitivity: float = 1.0, random_state=None):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.mech = Laplace(
            epsilon=epsilon,
            sensitivity=sensitivity,
            random_state=random_state,
        )

    def sample(self) -> float:
        return float(self.mech.randomise(0.0))


class ChanPrivateMisraGries:
    def __init__(self, stream, k, epsilon, universe=None, random_state=None):
        if k <= 0:
            raise ValueError("k must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.stream = stream
        self.k = k
        self.epsilon = epsilon
        self.universe = universe
        self.random_state = random_state

    def misra_gries(self):
        """
        Standard Misra-Gries.
        This version deletes zero counters immediately.
        """
        counters = {}

        for x in self.stream:
            if x in counters:
                counters[x] += 1

            elif len(counters) < self.k:
                counters[x] = 1

            else:
                keys_to_delete = []

                for key in list(counters.keys()):
                    counters[key] -= 1
                    if counters[key] == 0:
                        keys_to_delete.append(key)

                for key in keys_to_delete:
                    del counters[key]

        return counters

    def compute(self):
        counters = self.misra_gries()

        if self.universe is None:
            # For experiments, use observed universe.
            # For the formal algorithm, universe should be the full domain U.
            universe = set(self.stream)
        else:
            universe = set(self.universe)

        noise = DiffprivlibLaplaceNoise(
            epsilon=self.epsilon,
            sensitivity=float(self.k),
            random_state=self.random_state,
        )

        noisy_counts = {}

        for x in universe:
            base_count = counters.get(x, 0)
            noisy_counts[x] = base_count + noise.sample()

        top_k = heapq.nlargest(
            self.k,
            noisy_counts.items(),
            key=lambda item: item[1],
        )

        return dict(top_k)

class DiffprivlibLaplaceNoise:
    def __init__(self, epsilon: float, sensitivity: float = 1.0, random_state=None):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.mech = Laplace(
            epsilon=epsilon,
            sensitivity=sensitivity,
            random_state=random_state,
        )

    def sample(self) -> float:
        # randomise(0.0) returns 0 + Laplace noise
        return float(self.mech.randomise(0.0))