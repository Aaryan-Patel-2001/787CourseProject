from DP_Misra_Gries.Testing import Test, TestKSweep
from pathlib import Path
from collections import Counter

def load_baby_names_stream(folder_path="BabyNamesDataset", include_gender=False):
    """
    Load SSA-style baby name files into a stream/list.

    Each input row has format:
        Name,Gender,Frequency

    If include_gender=False:
        Mary,F,7065 -> ["Mary", "Mary", ..., "Mary"]

    If include_gender=True:
        Mary,F,7065 -> [("Mary", "F"), ("Mary", "F"), ..., ("Mary", "F")]

    Returns:
        list[str] or list[tuple[str, str]]
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    stream = []

    files = sorted(folder.glob("yob*.txt"))

    if not files:
        raise FileNotFoundError(f"No yob*.txt files found in {folder}")

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                name, gender, frequency_str = line.split(",")
                frequency = int(frequency_str)

                item = (name, gender) if include_gender else name

                stream.extend([item] * frequency)

    return stream


stream = load_baby_names_stream("BabyNamesDataset")[:10000000]
counts = Counter(stream)

print("stream length:", len(stream))
print("distinct observed names:", len(counts))
print("starting Experiment")


epsilons = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ks = [500, 1000, 2500]
delta = 1/(len(stream)*len(stream))
trials = 10
experiment_name = "BabyNames"
for k in ks:
    ####### [ Graphs to expect out of this: 4]
    # Regular AE 
    # Top-K MAE 
    # Recall@k 
    # Max Error 
    #######
    Test(stream, k, epsilons, delta, trials=trials, ExperimentName=experiment_name)


ks = [100, 250, 500, 1000, 2500, 5000, 6260]
epsilons=[0.05, 0.1, 0.5, 1.0]
for epsilon in epsilons:
    ####### [ Graphs to expect out of this: 4]
    # Regular AE 
    # Top-K MAE 
    # Recall@k 
    # Max Error 
    #######
    TestKSweep(stream, ks, epsilon, delta, trials=trials, ExperimentName=experiment_name)