
import torch
import torch.nn.functional as F
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections
import argparse
import warnings
import sklearn.exceptions

from utils import fix_randomness, starting_logs, AverageMeter
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from trainers.abstract_trainer import AbstractTrainer

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


class Test(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super().__init__(args)

        self.last_results = None
        self.best_results = None


    def test(self):

        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        last_results = pd.DataFrame(columns=results_columns)
        best_results = pd.DataFrame(columns=results_columns)

        # Trainer
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.scenario_log_dir = os.path.join(self.exp_log_dir, src_id + "_to_" + trg_id + "_run_" + str(run_id))

                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id)

                # Build model
                self.initialize_algorithm()

                # Load chechpoint 
                last_chk, best_chk = self.load_checkpoint(self.scenario_log_dir)

                # Testing the last model
                self.algorithm.network.load_state_dict(last_chk)
                self.evaluate(self.trg_test_dl)
                last_metrics = self.calculate_metrics()
                last_results = self.append_results_to_tables(last_results, f"{src_id}_to_{trg_id}", run_id,
                                                             last_metrics)
                

                # Testing the best model
                self.algorithm.network.load_state_dict(best_chk)
                self.evaluate(self.trg_test_dl)
                best_metrics = self.calculate_metrics()
                # Append results to tables
                best_results = self.append_results_to_tables(best_results, f"{src_id}_to_{trg_id}", run_id,
                                                             best_metrics)

        summary_last = {metric: np.mean(last_results[metric]) for metric in results_columns[2:]}
        summary_best = {metric: np.mean(best_results[metric]) for metric in results_columns[2:]}

        # Calculate and append mean and std to tables
        last_results = self.add_mean_std_table(last_results, results_columns)
        best_results = self.add_mean_std_table(best_results, results_columns)

        # Save tables to file if needed
        self.save_tables_to_file(last_results, 'last_results')
        self.save_tables_to_file(best_results, 'best_results')

        for summary_name, summary in [('Last', summary_last), ('Best', summary_best)]:
            for key, val in summary.items():
                print(f'{summary_name}: {key}\t: {val:2.4f}')
