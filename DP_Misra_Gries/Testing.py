from Implementation import DP_Misra_Gries 
from collections import Counter
import math
import statistics
import matplotlib.pyplot as plt


def exact_histogram(stream):
    """
    True non-private counts.
    """
    return Counter(stream)


def evaluate_once(stream, k, epsilon, delta):
    """
    Run DP Misra-Gries once and compare output to exact histogram.
    Missing private counts are treated as 0.
    """
    true_counts = exact_histogram(stream)

    dp_mg = DP_Misra_Gries(
        stream=stream,
        k=k,
        epsilon=epsilon,
        delta=delta,
    )

    private_counts = dp_mg.compute()

    # Evaluate over all true items in the stream.
    # If an item is missing from private_counts, its estimate is 0.
    errors = {}

    for key, true_count in true_counts.items():
        private_estimate = private_counts.get(key, 0.0)
        errors[key] = abs(private_estimate - true_count)

    mae = sum(errors.values()) / len(errors)

    rmse = math.sqrt(
        sum(error ** 2 for error in errors.values()) / len(errors)
    )

    max_error = max(errors.values())

    return {
        "epsilon": epsilon,
        "mae": mae,
        "rmse": rmse,
        "max_error": max_error,
        "num_reported_items": len(private_counts),
        "private_counts": private_counts,
        "errors": errors,
    }

def evaluate_epsilon_sweep(stream, k, epsilons, delta, trials=20):
    """
    Run DP Misra-Gries for multiple epsilon values.

    Returns one summary row per epsilon.
    """
    results = []

    for epsilon in epsilons:
        trial_results = []

        for _ in range(trials):
            result = evaluate_once(
                stream=stream,
                k=k,
                epsilon=epsilon,
                delta=delta,
            )
            trial_results.append(result)

        row = {
            "epsilon": epsilon,
            "avg_mae": statistics.mean(r["mae"] for r in trial_results),
            "avg_rmse": statistics.mean(r["rmse"] for r in trial_results),
            "avg_max_error": statistics.mean(r["max_error"] for r in trial_results),
            "avg_num_reported_items": statistics.mean(
                r["num_reported_items"] for r in trial_results
            ),
        }

        results.append(row)

    return results

def plot_epsilon_sweep(summary):
    epsilons = [row["epsilon"] for row in summary]
    maes = [row["avg_mae"] for row in summary]
    rmses = [row["avg_rmse"] for row in summary]
    max_errors = [row["avg_max_error"] for row in summary]

    plt.figure()
    plt.plot(epsilons, maes, marker="o", label="MAE")
    plt.plot(epsilons, rmses, marker="o", label="RMSE")
    plt.plot(epsilons, max_errors, marker="o", label="Max error")

    plt.xscale("log")
    plt.xlabel("epsilon")
    plt.ylabel("count error")
    plt.title("Private Misra-Gries accuracy vs epsilon")
    plt.legend()
    plt.show()


if __name__ == "__main__": 
    stream = (
    ["a"] * 500
    + ["b"] * 300
    + ["c"] * 150
    + ["d"] * 80
    + ["e"] * 50
    + ["f"] * 20
    + ["g"] * 10
    )
    k = 5
    delta = 1e-6

    epsilons = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

    summary = evaluate_epsilon_sweep(
        stream=stream,
        k=k,
        epsilons=epsilons,
        delta=delta,
        trials=50,
    )

    plot_epsilon_sweep(summary)

    for row in summary:
        print(row)