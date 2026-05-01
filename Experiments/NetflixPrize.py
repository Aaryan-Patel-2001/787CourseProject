
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from DP_Misra_Gries.Testing import Test, TestKSweep
from pathlib import Path
from collections import Counter

def load_netflix_stream(folder_path="netflix-prize-data"):
    """
    Load preprocessed Netflix review data into a stream/list.

    Each input row has format:
        MovieID, Date (dropped)

    Returns:
        list[str]
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    stream = []

    files = sorted(folder.glob("processed_stream_1.txt"))

    if not files:
        raise FileNotFoundError(f"No processed_stream_*.txt files found in {folder}")

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                movie_str, date = line.split(",")
                movie = int(movie_str)

                stream.append(movie)

    return stream


stream = load_netflix_stream("netflix-prize-data")
counts = Counter(stream)

print("stream length:", len(stream))
print("distinct observed movies:", len(counts))
print("starting Experiment")

k = 100
epsilons = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
delta = 1e-6 
trials = 3
experiment_name = "NetflixPrize"
Test(stream, k, epsilons, delta, trials=trials, ExperimentName=experiment_name)

ks = [5, 10, 50, 100, 150, 200, 500]
epsilon=1.0
#TestKSweep(stream, ks, epsilon, delta, trials=trials, ExperimentName=experiment_name)
