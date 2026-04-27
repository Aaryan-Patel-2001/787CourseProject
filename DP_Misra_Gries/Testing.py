from pathlib import Path

from DP_Misra_Gries.Implementation import DP_Misra_Gries, ChanPrivateMisraGries
from collections import Counter
import math
import statistics
import matplotlib.pyplot as plt

class Test: 
    def __init__(self, stream, k, epsilons, delta, universe=None, trials=20, ExperimentName=""):
        self.stream = stream 
        self.k = k 
        self.epsilons = epsilons 
        self.delta = delta 
        self.universe = universe
        self.trials = trials
        self.ExperimentName = ExperimentName

        pmg_summary = self.evaluate_epsilon_sweep("pmg")
        self.plot_pmg_only(pmg_summary)
        chan_summary = self.evaluate_epsilon_sweep("chan")

        self.plot_comparison(pmg_summary, chan_summary)

    def exact_histogram(self):
        """
        True non-private counts.
        """
        return Counter(self.stream)

    def evaluate_once(self, epsilon, algorithm):
        """
        Run one private Misra-Gries variant and compare output to exact histogram.
        Missing private counts are treated as 0.

        algorithm:
            "pmg"  -> new paper's DP_Misra_Gries
            "chan" -> Chan et al.-style private Misra-Gries
        """
        true_counts = self.exact_histogram()

        if algorithm == "pmg":
            model = DP_Misra_Gries(
                stream=self.stream,
                k=self.k,
                epsilon=epsilon,
                delta=self.delta,
            )

        elif algorithm == "chan":
            model = ChanPrivateMisraGries(
                stream=self.stream,
                k=self.k,
                epsilon=epsilon,
                universe=self.universe,
            )

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        private_counts = model.compute()

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
            "algorithm": algorithm,
            "epsilon": epsilon,
            "mae": mae,
            "rmse": rmse,
            "max_error": max_error,
            "num_reported_items": len(private_counts),
            "private_counts": private_counts,
            "errors": errors,
        }


    def evaluate_epsilon_sweep(self, algorithm):
        """
        Run one algorithm for multiple epsilon values.

        Returns one summary row per epsilon.
        """
        results = []

        for epsilon in self.epsilons:
            trial_results = []

            for _ in range(self.trials):
                result = self.evaluate_once(
                    epsilon=epsilon,
                    algorithm=algorithm,
                )
                trial_results.append(result)

            row = {
                "algorithm": algorithm,
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

    def plot_comparison(self, pmg_summary, chan_summary):
        pmg_epsilons = [row["epsilon"] for row in pmg_summary]
        chan_epsilons = [row["epsilon"] for row in chan_summary]

        pmg_maes = [row["avg_mae"] for row in pmg_summary]
        chan_maes = [row["avg_mae"] for row in chan_summary]

        pmg_rmses = [row["avg_rmse"] for row in pmg_summary]
        chan_rmses = [row["avg_rmse"] for row in chan_summary]

        pmg_max_errors = [row["avg_max_error"] for row in pmg_summary]
        chan_max_errors = [row["avg_max_error"] for row in chan_summary]

        plt.figure()
        plt.plot(pmg_epsilons, pmg_maes, marker="o", label="New PMG - MAE")
        plt.plot(chan_epsilons, chan_maes, marker="o", label="Chan MG - MAE")
        plt.xscale("log")
        plt.xlabel("epsilon")
        plt.ylabel("MAE")
        plt.title("MAE vs epsilon")
        plt.legend()
        fileName = f"MAE_{self.ExperimentName}_{self.k}_{self.delta}.pdf"
        plt.savefig(f"Fig/{fileName}")
        plt.show()

        plt.figure()
        plt.plot(pmg_epsilons, pmg_rmses, marker="o", label="New PMG - RMSE")
        plt.plot(chan_epsilons, chan_rmses, marker="o", label="Chan MG - RMSE")
        plt.xscale("log")
        plt.xlabel("epsilon")
        plt.ylabel("RMSE")
        plt.title("RMSE vs epsilon")
        plt.legend()
        fileName = f"RMSE_{self.ExperimentName}_{self.k}_{self.delta}.pdf"
        plt.savefig(f"Fig/{fileName}")
        plt.show()

        plt.figure()
        plt.plot(pmg_epsilons, pmg_max_errors, marker="o", label="New PMG - Max error")
        plt.plot(chan_epsilons, chan_max_errors, marker="o", label="Chan MG - Max error")
        plt.xscale("log")
        plt.xlabel("epsilon")
        plt.ylabel("Max error")
        plt.title("Max error vs epsilon")
        plt.legend()
        fileName = f"MaxError_{self.ExperimentName}_{self.k}_{self.delta}.pdf"
        plt.savefig(f"Fig/{fileName}")
        plt.show()

    def plot_pmg_only(self, pmg_summary):
        pmg_epsilons = [row["epsilon"] for row in pmg_summary]

        pmg_maes = [row["avg_mae"] for row in pmg_summary]
        pmg_rmses = [row["avg_rmse"] for row in pmg_summary]
        pmg_max_errors = [row["avg_max_error"] for row in pmg_summary]

        plt.figure()
        plt.plot(pmg_epsilons, pmg_maes, marker="o", label="New PMG - MAE")
        plt.xscale("log")
        plt.xlabel("epsilon")
        plt.ylabel("MAE")
        plt.title("PMG MAE vs epsilon")
        plt.legend()
        fileName = f"PMG_MAE_{self.ExperimentName}_{self.k}_{self.delta}.pdf"
        plt.savefig(f"Fig/{fileName}", bbox_inches="tight")
        plt.show()

        plt.figure()
        plt.plot(pmg_epsilons, pmg_rmses, marker="o", label="New PMG - RMSE")
        plt.xscale("log")
        plt.xlabel("epsilon")
        plt.ylabel("RMSE")
        plt.title("PMG RMSE vs epsilon")
        plt.legend()
        fileName = f"PMG_RMSE_{self.ExperimentName}_{self.k}_{self.delta}.pdf"
        plt.savefig(f"Fig/{fileName}", bbox_inches="tight")
        plt.show()

        plt.figure()
        plt.plot(pmg_epsilons, pmg_max_errors, marker="o", label="New PMG - Max error")
        plt.xscale("log")
        plt.xlabel("epsilon")
        plt.ylabel("Max error")
        plt.title("PMG Max error vs epsilon")
        plt.legend()
        fileName = f"PMG_MaxError_{self.ExperimentName}_{self.k}_{self.delta}.pdf"
        plt.savefig(f"Fig/{fileName}", bbox_inches="tight")
        plt.show()



class TestKSweep:
    def __init__(
        self,
        stream,
        ks,
        epsilon,
        delta,
        universe=None,
        trials=20,
        ExperimentName="",
        fig_dir="Fig",
    ):
        self.stream = stream
        self.ks = ks
        self.epsilon = epsilon
        self.delta = delta
        self.universe = universe
        self.trials = trials
        self.ExperimentName = ExperimentName
        self.fig_dir = Path(fig_dir)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.true_counts = self.exact_histogram()

        pmg_summary = self.evaluate_k_sweep("pmg")
        chan_summary = self.evaluate_k_sweep("chan")

        self.plot_comparison(pmg_summary, chan_summary)

    def exact_histogram(self):
        """
        True non-private counts.
        """
        return Counter(self.stream)

    def evaluate_once(self, k, algorithm):
        """
        Run one private Misra-Gries variant and compare output to exact histogram.
        Missing private counts are treated as 0.

        algorithm:
            "pmg"  -> new paper's DP_Misra_Gries
            "chan" -> Chan et al.-style private Misra-Gries
        """
        if algorithm == "pmg":
            model = DP_Misra_Gries(
                stream=self.stream,
                k=k,
                epsilon=self.epsilon,
                delta=self.delta,
            )

        elif algorithm == "chan":
            model = ChanPrivateMisraGries(
                stream=self.stream,
                k=k,
                epsilon=self.epsilon,
                universe=self.universe,
            )

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        private_counts = model.compute()

        errors = {}

        for key, true_count in self.true_counts.items():
            private_estimate = private_counts.get(key, 0.0)
            errors[key] = abs(private_estimate - true_count)

        mae = sum(errors.values()) / len(errors)

        rmse = math.sqrt(
            sum(error ** 2 for error in errors.values()) / len(errors)
        )

        max_error = max(errors.values())

        return {
            "algorithm": algorithm,
            "k": k,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "mae": mae,
            "rmse": rmse,
            "max_error": max_error,
            "num_reported_items": len(private_counts),
            "private_counts": private_counts,
            "errors": errors,
        }

    def evaluate_k_sweep(self, algorithm):
        """
        Run one algorithm for multiple k values.

        Returns one summary row per k.
        """
        results = []

        for k in self.ks:
            trial_results = []

            for _ in range(self.trials):
                result = self.evaluate_once(
                    k=k,
                    algorithm=algorithm,
                )
                trial_results.append(result)

            row = {
                "algorithm": algorithm,
                "k": k,
                "epsilon": self.epsilon,
                "delta": self.delta,
                "avg_mae": statistics.mean(r["mae"] for r in trial_results),
                "avg_rmse": statistics.mean(r["rmse"] for r in trial_results),
                "avg_max_error": statistics.mean(r["max_error"] for r in trial_results),
                "avg_num_reported_items": statistics.mean(
                    r["num_reported_items"] for r in trial_results
                ),
            }

            results.append(row)

        return results

    def plot_comparison(self, pmg_summary, chan_summary):
        pmg_ks = [row["k"] for row in pmg_summary]
        chan_ks = [row["k"] for row in chan_summary]

        pmg_maes = [row["avg_mae"] for row in pmg_summary]
        chan_maes = [row["avg_mae"] for row in chan_summary]

        pmg_rmses = [row["avg_rmse"] for row in pmg_summary]
        chan_rmses = [row["avg_rmse"] for row in chan_summary]

        pmg_max_errors = [row["avg_max_error"] for row in pmg_summary]
        chan_max_errors = [row["avg_max_error"] for row in chan_summary]

        # MAE plot
        plt.figure()
        plt.plot(pmg_ks, pmg_maes, marker="o", label="New PMG - MAE")
        plt.plot(chan_ks, chan_maes, marker="o", label="Chan MG - MAE")
        plt.xlabel("k")
        plt.ylabel("MAE")
        plt.title(f"MAE vs k, epsilon={self.epsilon}, delta={self.delta}")
        plt.legend()
        file_name = f"MAE_vs_k_{self.ExperimentName}_eps{self.epsilon}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")
        plt.show()

        # RMSE plot
        plt.figure()
        plt.plot(pmg_ks, pmg_rmses, marker="o", label="New PMG - RMSE")
        plt.plot(chan_ks, chan_rmses, marker="o", label="Chan MG - RMSE")
        plt.xlabel("k")
        plt.ylabel("RMSE")
        plt.title(f"RMSE vs k, epsilon={self.epsilon}, delta={self.delta}")
        plt.legend()
        file_name = f"RMSE_vs_k_{self.ExperimentName}_eps{self.epsilon}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")
        plt.show()

        # Max error plot
        plt.figure()
        plt.plot(pmg_ks, pmg_max_errors, marker="o", label="New PMG - Max error")
        plt.plot(chan_ks, chan_max_errors, marker="o", label="Chan MG - Max error")
        plt.xlabel("k")
        plt.ylabel("Max error")
        plt.title(f"Max Error vs k, epsilon={self.epsilon}, delta={self.delta}")
        plt.legend()
        file_name = f"MaxError_vs_k_{self.ExperimentName}_eps{self.epsilon}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")
        plt.show()

    def plot_pmg_only(self, pmg_summary):
        pmg_ks = [row["k"] for row in pmg_summary]

        pmg_maes = [row["avg_mae"] for row in pmg_summary]
        pmg_rmses = [row["avg_rmse"] for row in pmg_summary]
        pmg_max_errors = [row["avg_max_error"] for row in pmg_summary]

        # MAE plot
        plt.figure()
        plt.plot(pmg_ks, pmg_maes, marker="o", label="New PMG - MAE")
        plt.xlabel("k")
        plt.ylabel("MAE")
        plt.title(f"PMG MAE vs k, epsilon={self.epsilon}, delta={self.delta}")
        plt.legend()
        file_name = f"PMG_MAE_vs_k_{self.ExperimentName}_eps{self.epsilon}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")
        plt.show()

        # RMSE plot
        plt.figure()
        plt.plot(pmg_ks, pmg_rmses, marker="o", label="New PMG - RMSE")
        plt.xlabel("k")
        plt.ylabel("RMSE")
        plt.title(f"PMG RMSE vs k, epsilon={self.epsilon}, delta={self.delta}")
        plt.legend()
        file_name = f"PMG_RMSE_vs_k_{self.ExperimentName}_eps{self.epsilon}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")
        plt.show()

        # Max error plot
        plt.figure()
        plt.plot(pmg_ks, pmg_max_errors, marker="o", label="New PMG - Max error")
        plt.xlabel("k")
        plt.ylabel("Max error")
        plt.title(f"PMG Max Error vs k, epsilon={self.epsilon}, delta={self.delta}")
        plt.legend()
        file_name = f"PMG_MaxError_vs_k_{self.ExperimentName}_eps{self.epsilon}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")
        plt.show()