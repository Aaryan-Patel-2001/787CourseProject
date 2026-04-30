from pathlib import Path
from collections import Counter
import math
import statistics
import matplotlib.pyplot as plt

from DP_Misra_Gries.Implementation import DP_Misra_Gries, ChanPrivateMisraGries


class Test:
    def __init__(
        self,
        stream,
        k,
        epsilons,
        delta,
        universe=None,
        trials=20,
        ExperimentName="",
        fig_dir="Fig",
    ):
        self.stream = stream
        self.k = k
        self.epsilons = epsilons
        self.delta = delta
        self.universe = universe
        self.trials = trials
        self.ExperimentName = ExperimentName

        k_dir_name = f"k_{self.k}"
        self.fig_dir = Path(fig_dir) / k_dir_name
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        # self.fig_dir = Path(fig_dir)
        # self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.true_counts = self.exact_histogram()

        pmg_summary = self.evaluate_epsilon_sweep("pmg")
        chan_summary = self.evaluate_epsilon_sweep("chan")

        self.plot_pmg_only(pmg_summary)
        self.plot_comparison(pmg_summary, chan_summary)

        print("\nPMG summary:")
        for row in pmg_summary:
            print(row)

        print("\nChan summary:")
        for row in chan_summary:
            print(row)

    def exact_histogram(self):
        return Counter(self.stream)

    def true_top_k_keys(self, k):
        return {key for key, _ in self.true_counts.most_common(k)}

    def compute_metrics(self, private_counts, k):
        """
        Metrics:
            regular MAE: over all observed stream keys
            RMSE: over all observed stream keys
            max error: over all observed stream keys
            top-k MAE: over true top-k keys only
            recall@k: fraction of true top-k keys returned
        """
        errors = {}

        for key, true_count in self.true_counts.items():
            private_estimate = private_counts.get(key, 0.0)
            errors[key] = abs(private_estimate - true_count)

        mae = sum(errors.values()) / len(errors)

        rmse = math.sqrt(
            sum(error ** 2 for error in errors.values()) / len(errors)
        )

        max_error = max(errors.values())

        true_top_k = self.true_top_k_keys(k)
        output_keys = set(private_counts.keys())

        top_k_errors = []
        for key in true_top_k:
            true_count = self.true_counts[key]
            private_estimate = private_counts.get(key, 0.0)
            top_k_errors.append(abs(private_estimate - true_count))

        top_k_mae = sum(top_k_errors) / len(top_k_errors)

        recall_at_k = len(output_keys & true_top_k) / len(true_top_k)

        return {
            "mae": mae,
            "rmse": rmse,
            "max_error": max_error,
            "top_k_mae": top_k_mae,
            "recall_at_k": recall_at_k,
            "errors": errors,
        }

    def evaluate_once(self, epsilon, algorithm):
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

        metrics = self.compute_metrics(
            private_counts=private_counts,
            k=self.k,
        )

        return {
            "algorithm": algorithm,
            "epsilon": epsilon,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "max_error": metrics["max_error"],
            "top_k_mae": metrics["top_k_mae"],
            "recall_at_k": metrics["recall_at_k"],
            "num_reported_items": len(private_counts),
            "private_counts": private_counts,
            "errors": metrics["errors"],
        }

    def evaluate_epsilon_sweep(self, algorithm):
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
                "avg_top_k_mae": statistics.mean(r["top_k_mae"] for r in trial_results),
                "avg_recall_at_k": statistics.mean(r["recall_at_k"] for r in trial_results),
                "avg_num_reported_items": statistics.mean(
                    r["num_reported_items"] for r in trial_results
                ),
            }

            results.append(row)

        return results

    def plot_pmg_only(self, pmg_summary):
        epsilons = [row["epsilon"] for row in pmg_summary]

        regular_mae = [row["avg_mae"] for row in pmg_summary]
        top_k_mae = [row["avg_top_k_mae"] for row in pmg_summary]
        recall_at_k = [row["avg_recall_at_k"] for row in pmg_summary]
        num_reported = [row["avg_num_reported_items"] for row in pmg_summary]

        plt.figure()
        plt.plot(epsilons, regular_mae, marker="o", label="PMG - Regular MAE")
        plt.xscale("log")
        plt.xlabel("epsilon")
        plt.ylabel("MAE")
        plt.title("PMG Regular MAE vs epsilon")
        plt.legend()
        file_name = f"PMG_MAE{self.ExperimentName}_k{self.k}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")

        plt.figure()
        plt.plot(epsilons, top_k_mae, marker="o", label="PMG - Top-k MAE")
        plt.xscale("log")
        plt.xlabel("epsilon")
        plt.ylabel("MAE")
        plt.title("PMG Top-k MAE vs epsilon")
        plt.legend()
        file_name = f"PMG_TopKMAE_{self.ExperimentName}_k{self.k}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")

        plt.figure()
        plt.plot(epsilons, recall_at_k, marker="o", label="PMG - Recall@k")
        plt.xscale("log")
        plt.xlabel("epsilon")
        plt.ylabel("Recall@k")
        plt.ylim(0, 1.05)
        plt.title("PMG Recall@k vs epsilon")
        plt.legend()
        file_name = f"PMG_RecallAtK_{self.ExperimentName}_k{self.k}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")

        plt.figure()
        plt.plot(epsilons, num_reported, marker="o", label="PMG - Output size")
        plt.xscale("log")
        plt.xlabel("epsilon")
        plt.ylabel("Number of reported items")
        plt.title("PMG Output Size vs epsilon")
        plt.legend()
        file_name = f"PMG_OutputSize_{self.ExperimentName}_k{self.k}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")

    def plot_comparison(self, pmg_summary, chan_summary):
        pmg_epsilons = [row["epsilon"] for row in pmg_summary]
        chan_epsilons = [row["epsilon"] for row in chan_summary]

        pmg_regular_mae = [row["avg_mae"] for row in pmg_summary]
        chan_regular_mae = [row["avg_mae"] for row in chan_summary]

        pmg_top_k_mae = [row["avg_top_k_mae"] for row in pmg_summary]
        chan_top_k_mae = [row["avg_top_k_mae"] for row in chan_summary]

        pmg_recall = [row["avg_recall_at_k"] for row in pmg_summary]
        chan_recall = [row["avg_recall_at_k"] for row in chan_summary]

        pmg_max_error = [row["avg_max_error"] for row in pmg_summary]
        chan_max_error = [row["avg_max_error"] for row in chan_summary]

        plt.figure()
        plt.plot(pmg_epsilons, pmg_regular_mae, marker="o", label="New PMG - Regular MAE")
        plt.plot(chan_epsilons, chan_regular_mae, marker="o", label="Chan MG - Regular MAE")
        plt.xscale("log")
        plt.xlabel("epsilon")
        plt.ylabel("Regular MAE")
        plt.title("Regular MAE vs epsilon")
        plt.legend()
        file_name = f"RegularMAE_{self.ExperimentName}_k{self.k}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")

        plt.figure()
        plt.plot(pmg_epsilons, pmg_top_k_mae, marker="o", label="New PMG - Top-k MAE")
        plt.plot(chan_epsilons, chan_top_k_mae, marker="o", label="Chan MG - Top-k MAE")
        plt.xscale("log")
        plt.xlabel("epsilon")
        plt.ylabel("Top-k MAE")
        plt.title("Top-k MAE vs epsilon")
        plt.legend()
        file_name = f"TopKMAE_{self.ExperimentName}_k{self.k}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")

        plt.figure()
        plt.plot(pmg_epsilons, pmg_recall, marker="o", label="New PMG - Recall@k")
        plt.plot(chan_epsilons, chan_recall, marker="o", label="Chan MG - Recall@k")
        plt.xscale("log")
        plt.xlabel("epsilon")
        plt.ylabel("Recall@k")
        plt.ylim(0, 1.05)
        plt.title("Recall@k vs epsilon")
        plt.legend()
        file_name = f"RecallAtK_{self.ExperimentName}_k{self.k}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")

        plt.figure()
        plt.plot(pmg_epsilons, pmg_max_error, marker="o", label="New PMG - Max Error")
        plt.plot(chan_epsilons, chan_max_error, marker="o", label="Chan MG - Max Error")
        plt.xscale("log")
        plt.xlabel("epsilon")
        plt.ylabel("Max Error")
        plt.title("Max Error vs epsilon")
        plt.legend()
        file_name = f"MaxError_{self.ExperimentName}_k{self.k}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")


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

        epsilon_dir_name = f"epsilon_{self.epsilon}"
        self.fig_dir = Path(fig_dir) / epsilon_dir_name
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.true_counts = self.exact_histogram()

        pmg_summary = self.evaluate_k_sweep("pmg")
        chan_summary = self.evaluate_k_sweep("chan")

        self.plot_pmg_only(pmg_summary)
        self.plot_comparison(pmg_summary, chan_summary)

        print("\nPMG k-sweep summary:")
        for row in pmg_summary:
            print(row)

        print("\nChan k-sweep summary:")
        for row in chan_summary:
            print(row)

    def exact_histogram(self):
        return Counter(self.stream)

    def true_top_k_keys(self, k):
        return {key for key, _ in self.true_counts.most_common(k)}

    def compute_metrics(self, private_counts, k):
        errors = {}

        for key, true_count in self.true_counts.items():
            private_estimate = private_counts.get(key, 0.0)
            errors[key] = abs(private_estimate - true_count)

        mae = sum(errors.values()) / len(errors)

        rmse = math.sqrt(
            sum(error ** 2 for error in errors.values()) / len(errors)
        )

        max_error = max(errors.values())

        true_top_k = self.true_top_k_keys(k)
        output_keys = set(private_counts.keys())

        top_k_errors = []
        for key in true_top_k:
            true_count = self.true_counts[key]
            private_estimate = private_counts.get(key, 0.0)
            top_k_errors.append(abs(private_estimate - true_count))

        top_k_mae = sum(top_k_errors) / len(top_k_errors)

        recall_at_k = len(output_keys & true_top_k) / len(true_top_k)

        return {
            "mae": mae,
            "rmse": rmse,
            "max_error": max_error,
            "top_k_mae": top_k_mae,
            "recall_at_k": recall_at_k,
            "errors": errors,
        }

    def evaluate_once(self, k, algorithm):
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

        metrics = self.compute_metrics(
            private_counts=private_counts,
            k=k,
        )

        return {
            "algorithm": algorithm,
            "k": k,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "max_error": metrics["max_error"],
            "top_k_mae": metrics["top_k_mae"],
            "recall_at_k": metrics["recall_at_k"],
            "num_reported_items": len(private_counts),
            "private_counts": private_counts,
            "errors": metrics["errors"],
        }

    def evaluate_k_sweep(self, algorithm):
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
                "avg_top_k_mae": statistics.mean(r["top_k_mae"] for r in trial_results),
                "avg_recall_at_k": statistics.mean(r["recall_at_k"] for r in trial_results),
                "avg_num_reported_items": statistics.mean(
                    r["num_reported_items"] for r in trial_results
                ),
            }

            results.append(row)

        return results

    def plot_pmg_only(self, pmg_summary):
        ks = [row["k"] for row in pmg_summary]

        regular_mae = [row["avg_mae"] for row in pmg_summary]
        top_k_mae = [row["avg_top_k_mae"] for row in pmg_summary]
        recall_at_k = [row["avg_recall_at_k"] for row in pmg_summary]
        num_reported = [row["avg_num_reported_items"] for row in pmg_summary]

        plt.figure()
        plt.plot(ks, regular_mae, marker="o", label="PMG - Regular MAE")
        plt.xlabel("k")
        plt.ylabel("MAE")
        plt.title(f"PMG Regular MAE vs k, epsilon={self.epsilon}")
        plt.legend()
        file_name = f"PMG_MAE_vs_k_{self.ExperimentName}_eps{self.epsilon}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")

        plt.figure()
        plt.plot(ks, top_k_mae, marker="o", label="PMG - Top-k MAE")
        plt.xlabel("k")
        plt.ylabel("MAE")
        plt.title(f"PMG Top-k MAE vs k, epsilon={self.epsilon}")
        plt.legend()
        file_name = f"PMG_TopKMAE_vs_k_{self.ExperimentName}_eps{self.epsilon}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")

        plt.figure()
        plt.plot(ks, recall_at_k, marker="o", label="PMG - Recall@k")
        plt.xlabel("k")
        plt.ylabel("Recall@k")
        plt.ylim(0, 1.05)
        plt.title(f"PMG Recall@k vs k, epsilon={self.epsilon}")
        plt.legend()
        file_name = f"PMG_RecallAtK_vs_k_{self.ExperimentName}_eps{self.epsilon}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")

        plt.figure()
        plt.plot(ks, num_reported, marker="o", label="PMG - Output size")
        plt.xlabel("k")
        plt.ylabel("Number of reported items")
        plt.title(f"PMG Output Size vs k, epsilon={self.epsilon}")
        plt.legend()
        file_name = f"PMG_OutputSize_vs_k_{self.ExperimentName}_eps{self.epsilon}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")

    def plot_comparison(self, pmg_summary, chan_summary):
        pmg_ks = [row["k"] for row in pmg_summary]
        chan_ks = [row["k"] for row in chan_summary]

        pmg_regular_mae = [row["avg_mae"] for row in pmg_summary]
        chan_regular_mae = [row["avg_mae"] for row in chan_summary]

        pmg_top_k_mae = [row["avg_top_k_mae"] for row in pmg_summary]
        chan_top_k_mae = [row["avg_top_k_mae"] for row in chan_summary]

        pmg_recall = [row["avg_recall_at_k"] for row in pmg_summary]
        chan_recall = [row["avg_recall_at_k"] for row in chan_summary]

        pmg_max_error = [row["avg_max_error"] for row in pmg_summary]
        chan_max_error = [row["avg_max_error"] for row in chan_summary]

        plt.figure()
        plt.plot(pmg_ks, pmg_regular_mae, marker="o", label="New PMG - Regular MAE")
        plt.plot(chan_ks, chan_regular_mae, marker="o", label="Chan MG - Regular MAE")
        plt.xlabel("k")
        plt.ylabel("Regular MAE")
        plt.title(f"Regular MAE vs k, epsilon={self.epsilon}")
        plt.legend()
        file_name = f"RegularMAE_vs_k_{self.ExperimentName}_eps{self.epsilon}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")

        plt.figure()
        plt.plot(pmg_ks, pmg_top_k_mae, marker="o", label="New PMG - Top-k MAE")
        plt.plot(chan_ks, chan_top_k_mae, marker="o", label="Chan MG - Top-k MAE")
        plt.xlabel("k")
        plt.ylabel("Top-k MAE")
        plt.title(f"Top-k MAE vs k, epsilon={self.epsilon}")
        plt.legend()
        file_name = f"TopKMAE_vs_k_{self.ExperimentName}_eps{self.epsilon}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")

        plt.figure()
        plt.plot(pmg_ks, pmg_recall, marker="o", label="New PMG - Recall@k")
        plt.plot(chan_ks, chan_recall, marker="o", label="Chan MG - Recall@k")
        plt.xlabel("k")
        plt.ylabel("Recall@k")
        plt.ylim(0, 1.05)
        plt.title(f"Recall@k vs k, epsilon={self.epsilon}")
        plt.legend()
        file_name = f"RecallAtK_vs_k_{self.ExperimentName}_eps{self.epsilon}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")

        plt.figure()
        plt.plot(pmg_ks, pmg_max_error, marker="o", label="New PMG - Max Error")
        plt.plot(chan_ks, chan_max_error, marker="o", label="Chan MG - Max Error")
        plt.xlabel("k")
        plt.ylabel("Max Error")
        plt.title(f"Max Error vs k, epsilon={self.epsilon}")
        plt.legend()
        file_name = f"MaxError_vs_k_{self.ExperimentName}_eps{self.epsilon}_delta{self.delta}.pdf"
        plt.savefig(self.fig_dir / file_name, bbox_inches="tight")