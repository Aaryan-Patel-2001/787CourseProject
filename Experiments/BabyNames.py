from DP_Misra_Gries.Testing import Test
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


stream = load_baby_names_stream("BabyNamesDataset")
stream = stream[:1000000]
counts = Counter(stream)

print("stream length:", len(stream))
print("distinct observed names:", len(counts))
print("starting Experiment")

k = 50
epsilons = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
delta = 1e-6 
trials = 3
experiment_name = "BabyNames"

Test(stream, k, epsilons, delta, trials=trials, ExperimentName=experiment_name)