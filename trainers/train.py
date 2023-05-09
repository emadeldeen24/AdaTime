import sys
sys.path.append('../ADATIME')

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
from abstract_trainer import AbstractTrainer
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()
       


class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # Logging
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description, f"{self.run_description}")
        os.makedirs(self.exp_log_dir, exist_ok=True)


    def train(self):

        # table with metrics
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = pd.DataFrame(columns=results_columns)

        # table with risks
        risks_columns = ["scenario", "run", "src_risk", "few_shot_risk", "trg_risk"]
        table_risks = pd.DataFrame(columns=risks_columns)


        # Trainer
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                src_id, trg_id, run_id)
                # Average meters
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Load data
                self.load_data(src_id, trg_id)

                # Train model
                last_model, best_model = self.train_model()

                # Save checkpoint
                self.save_checkpoint(self.home_path, self.scenario_log_dir, last_model, best_model)

                # Calculate risks and metrics
                metrics = self.calculate_metrics()
                risks = self.calculate_risks()

                # Append results to tables
                scenario = f"{src_id}_to_{trg_id}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics)
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, risks)

        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, results_columns)
        table_risks = self.add_mean_std_table(table_risks, risks_columns)


        # Save tables to file if needed
        self.save_tables_to_file(table_results, 'results')
        self.save_tables_to_file(table_risks, 'risks')


if __name__ == "__main__":

    # ========  Experiments Name ================
    parser.add_argument('--save_dir',               default='../experiments_logs',         type=str, help='Directory containing all experiments')
    
    # ========= Select the DA methods ============
    parser.add_argument('--da_method',              default='Deep_Coral',               type=str, help='DANN, Deep_Coral, WDGRL, MMDA, VADA, DIRT, CDAN, ADDA, HoMM, CoDATS')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path',              default=r'../ADATIME_data',                  type=str, help='Path containing datase2t')
    parser.add_argument('--dataset',                default='HAR',                      type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone',               default='CNN',                      type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs',               default=3,                          type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device',                 default= "cuda",                   type=str, help='cpu or cuda')

    args = parser.parse_args()


    trainer = Trainer(args)
    trainer.train()