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
       


class Tester(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super(Tester, self).__init__(args)

        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, self.run_description)

    def load_checkpoint(self, model_dir):
        checkpoint = torch.load(model_dir)
        last_model = checkpoint['last_model']
        best_model = checkpoint['best_model']
        return last_model, best_model
        
    def build_model(self):
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        return algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device).to(self.device)

    def test(self):

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
                
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())


                # Load data
                self.load_data(src_id, trg_id)

                
                # Build model
                self.algorithm = self.build_model()

                # load chechpoint 
                last_chk, best_chk = self.load_checkpoint( self.scenario_log_dir)

                # load the model dictionary 
                self.algorithm.network.load_state_dict(best_chk)


                # testing the model
                self.test_model(self.algorithm)


                # Calculate risks and metrics
                risks, metrics = self.calculate_metrics_risks()

                # Append results to tables
                scenario = f"{src_id}_to_{trg_id}"
                table_results, table_risks = self.append_results_to_tables(table_results, table_risks, scenario, run_id, metrics, risks)

        # Calculate and append mean and std to tables
        table_results, table_risks = self.append_mean_std_to_tables(table_results, table_risks, results_columns, risks_columns)

        # Save tables to file if needed
        self.save_tables_to_file(table_results, table_risks)
     

    def test_model(self):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels in self.trg_test_dl:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = feature_extractor(data)
                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.item())
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.loss = torch.tensor(total_loss).mean()  # average loss
        self.full_preds = torch.cat((preds_list))
        self.full_labels = torch.cat((labels_list))

if __name__ == "__main__":

    # ========  Experiments Name ================
    parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
    
    # ========= Select the DA methods ============
    parser.add_argument('--da_method',              default='Deep_Coral',               type=str, help='DANN, Deep_Coral, WDGRL, MMDA, VADA, DIRT, CDAN, ADDA, HoMM, CoDATS')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path',              default=r'./data',                  type=str, help='Path containing datase2t')
    parser.add_argument('--dataset',                default='HAR',                      type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone',               default='CNN',                      type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs',               default=1,                          type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device',                 default= "mps",                   type=str, help='cpu or cuda')

    args = parser.parse_args()


    trainer = Trainer(args)
    trainer.train()