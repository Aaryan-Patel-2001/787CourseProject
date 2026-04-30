
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from DP_Misra_Gries.Testing import Test, TestKSweep
from pathlib import Path
from collections import Counter
from scapy.all import rdpcap

def load_caida_stream(folder_path="caida_passive_oc48"):
    """
    Load preprocessed

    Each input is an IP header. We extract the source IP address.

    Returns:
        list[int]
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    stream = []

    files = sorted(folder.glob("oc48-mfn.dirA.20030424-070000.UTC.anon.pcap"))

    if not files:
        raise FileNotFoundError(f"No files found in {folder}")

    for file_path in files:
        s = rdpcap(str(file_path))

        for pkt in s:
            p_bytes = pkt.load

            # basic sanity checks
            if len(p_bytes) < 24:
                continue
            if p_bytes[2:4] != b'\x08\x00':
                continue
            if p_bytes[4] >> 4 != 4:
                continue

            # src IP address
            src_ip = (p_bytes[16] << 24) + (p_bytes[17] << 16) + (p_bytes[18] << 8) + p_bytes[19]
            
            stream.append(src_ip)

    return stream


stream = load_caida_stream("caida_passive_oc48")
counts = Counter(stream)

print("stream length:", len(stream))
print("distinct observed source IPs:", len(counts))
print("starting Experiment")

k = 100
epsilons = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
delta = 1e-6 
trials = 3
experiment_name = "CaidaOC48"
Test(stream, k, epsilons, delta, trials=trials, ExperimentName=experiment_name)

ks = [5, 10, 50, 100, 150, 200, 500]
epsilon=1.0
#TestKSweep(stream, ks, epsilon, delta, trials=trials, ExperimentName=experiment_name)
