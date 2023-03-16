import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, F1Score
import os
import wandb
import pandas as pd
import numpy as np
from dataloader.dataloader import data_generator, few_shot_data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class

from configs.sweep_params import sweep_alg_hparams
from utils import fix_randomness, starting_logs, DictAsObject
import warnings
from sklearn.metrics import f1_score
import sklearn.exceptions

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

import collections
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from utils import AverageMeter, save_checkpoint

class AbstractTrainer(object):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device
        self.num_sweeps = args.num_sweeps

        # Exp Description
        self.run_description = args.run_description
        self.experiment_description = args.experiment_description
        # sweep parameters
        self.is_sweep = args.is_sweep
        self.sweep_project_wandb = args.sweep_project_wandb
        self.wandb_entity = args.wandb_entity
        self.hp_search_strategy = args.hp_search_strategy
        self.metric_to_minimize = args.metric_to_minimize

        # paths
        self.home_path = os.getcwd()
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.create_save_dir()

        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels

        # Specify number of hparams
        self.hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}

        # metrics

    def sweep(self):
        # sweep configurations
        sweep_runs_count = self.num_sweeps
        sweep_config = {
            'method': self.hp_search_strategy,
            'metric': {'name': self.metric_to_minimize, 'goal': 'minimize'},
            'name': self.da_method + '_' + self.backbone,
            'parameters': {**sweep_alg_hparams[self.da_method]}
        }
        sweep_id = wandb.sweep(sweep_config, project=self.sweep_project_wandb, entity=self.wandb_entity)

        wandb.agent(sweep_id, self.train, count=sweep_runs_count)  # Training with sweep
        # resuming sweep
        # wandb.agent('8wkaibgr', self.train, count=25,project='HHAR_SA_Resnet', entity= 'iclr_rebuttal' )

    def wandb_train(self):

        run = wandb.init(config=self.default_hparams)
        run_name = f"sweep_{self.dataset}"

        self.hparams = wandb.config

        # Logging
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, run_name)
        os.makedirs(self.exp_log_dir, exist_ok=True)

        # table with metrics
        table_results = wandb.Table(columns=["scenario", "run", "acc", "f1_score", "auroc"], allow_mixed_types=True)

        # table with risks
        table_risks = wandb.Table(columns=["scenario", "run", "src_risk", "few_shot_risk", "trg_risk"],
                                  allow_mixed_types=True)

        # metrics
        num_classes = self.dataset_configs.num_classes
        self.ACC = Accuracy(task="multiclass", num_classes=num_classes)  # .to(self.device)
        self.F1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")  # .to(self.device)
        self.AUROC = AUROC(task="multiclass", num_classes=num_classes)  # .to(self.device)

        # Trainer
        table_results, table_risks = self.cross_domain_trainer(table_results, table_risks)

        average_metrics = [np.mean(table_results.get_column(metric)) for metric in table_results.columns[2:]]
        std_metrics = [np.std(table_results.get_column(metric)) for metric in table_results.columns[2:]]

        # add avg and std values
        table_results.add_data('mean', '-', *average_metrics)
        table_results.add_data('std', '-', *std_metrics)

        # calculate overall
        overall_risks = {risk: np.mean(table_risks.get_column(risk)) for risk in table_risks.columns[2:]}
        overall_metrics = {metric: np.mean(table_results.get_column(metric)) for metric in table_results.columns[2:]}

        # log wabdb
        wandb.log({'results': table_results})
        wandb.log({'hparams': wandb.Table(
            dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']),
            allow_mixed_types=True)})
        wandb.log(overall_risks)
        wandb.log(overall_metrics)

        run.finish()

    def evaluate(self, test_loader):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
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

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id):
        self.src_train_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "train")
        self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "test")

        self.trg_train_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "train")
        self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "test")

        self.few_shot_dl_5 = few_shot_data_generator(self.trg_test_dl, self.dataset_configs,
                                                     5)  # set 5 to other value if you want other k-shot FST

    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def calculate_metrics_risks(self):
        # calculation based source test data
        self.evaluate(self.src_test_dl)
        src_risk = self.loss.item()
        # calculation based few_shot test data
        self.evaluate(self.few_shot_dl_5)
        fst_risk = self.loss.item()
        # calculation based target test data
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        # calculate metrics
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # f1_torch
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()
        # f1_sk learn
        # f1 = f1_score(self.full_preds.argmax(dim=1).cpu().numpy(), self.full_labels.cpu().numpy(), average='macro')

        risks = src_risk, fst_risk, trg_risk
        metrics = acc, f1, auroc

        return risks, metrics

class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

    def sweep(self):
        # sweep configurations
        sweep_runs_count = self.num_sweeps
        sweep_config = {
            'method': self.hp_search_strategy,
            'metric': {'name': self.metric_to_minimize, 'goal': 'minimize'},
            'name': self.da_method + '_' + self.backbone,
            'parameters': {**sweep_alg_hparams[self.da_method]}
        }
        sweep_id = wandb.sweep(sweep_config, project=self.sweep_project_wandb, entity=self.wandb_entity)

        wandb.agent(sweep_id, self.wandb_train, count=sweep_runs_count)
    def wandb_train(self):

        run = wandb.init(config=self.hparams)
        run_name = f"sweep_{self.dataset}"


        # Logging
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, run_name)
        os.makedirs(self.exp_log_dir, exist_ok=True)

        # table with metrics
        table_results = wandb.Table(columns=["scenario", "run", "acc", "f1_score", "auroc"], allow_mixed_types=True)

        # table with risks
        table_risks = wandb.Table(columns=["scenario", "run", "src_risk", "few_shot_risk", "trg_risk"],   allow_mixed_types=True)

        # metrics
        num_classes = self.dataset_configs.num_classes
        self.ACC = Accuracy(task="multiclass", num_classes=num_classes)  # .to(self.device)
        self.F1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")  # .to(self.device)
        self.AUROC = AUROC(task="multiclass", num_classes=num_classes)  # .to(self.device)

        # Trainer
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):  # specify number of consecutive runs
                # fixing random seed
                fix_randomness(run_id)
                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir, src_id, trg_id, run_id)

                # Load data
                self.load_data(src_id, trg_id)

                # get the backbone network
                backbone_fe = get_backbone_class(self.backbone)

                # get algorithm
                algorithm_class = get_algorithm_class(self.da_method)
                self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
                self.algorithm.to(self.device)

                # Average meters
                loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # run algorithm
                self.algorithm.update(self.src_train_dl, self.trg_train_dl, loss_avg_meters, self.logger)

                # calculate risks and metrics
                risks, metrics = self.calculate_metrics_risks()

                # calculate metrics on target test labeled data
                scenario = f"{src_id}_to_{trg_id}"

                # append results
                table_results.add_data(scenario, run_id, *metrics)
                table_risks.add_data(scenario, run_id, *risks)

        # average ,etrocs
        average_metrics = [np.mean(table_results.get_column(metric)) for metric in table_results.columns[2:]]
        std_metrics = [np.std(table_results.get_column(metric)) for metric in table_results.columns[2:]]

        # add avg and std values
        table_results.add_data('mean', '-', *average_metrics)
        table_results.add_data('std', '-', *std_metrics)

        # calculate overall
        overall_risks = {risk: np.mean(table_risks.get_column(risk)) for risk in table_risks.columns[2:]}
        overall_metrics = {metric: np.mean(table_results.get_column(metric)) for metric in table_results.columns[2:]}

        # log wandb
        wandb.log({'results': table_results})
        wandb.log({'hparams': wandb.Table(
            dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']),
            allow_mixed_types=True)})
        wandb.log(overall_risks)
        wandb.log(overall_metrics)

        # finish the run
        run.finish()
    def train(self):

        run_name = f"{self.run_description}"

        # Logging
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, run_name)
        os.makedirs(self.exp_log_dir, exist_ok=True)

        # table with metrics
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = pd.DataFrame(columns=results_columns)

        # table with risks
        risks_columns = ["scenario", "run", "src_risk", "few_shot_risk", "trg_risk"]
        table_risks = pd.DataFrame(columns=risks_columns)

        # metrics
        num_classes = self.dataset_configs.num_classes
        self.ACC = Accuracy(task="multiclass", num_classes=num_classes)  # .to(self.device)
        self.F1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")  # .to(self.device)
        self.AUROC = AUROC(task="multiclass", num_classes=num_classes)  # .to(self.device)

        # Trainer
        for src_id, trg_id in self.dataset_configs.scenarios:
            for run_id in range(self.num_runs):  # specify number of consecutive runs
                # fixing random seed
                fix_randomness(run_id)
                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)

                # Load data
                self.load_data(src_id, trg_id)

                # get algorithm
                algorithm_class = get_algorithm_class(self.da_method)
                backbone_fe = get_backbone_class(self.backbone)

                self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
                self.algorithm.to(self.device)

                # Average meters
                loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # run algorithm
                self.algorithm.update(self.src_train_dl, self.trg_train_dl, loss_avg_meters, self.logger)

                save_checkpoint(self.home_path, self.algorithm,  self.scenario_log_dir, type='last')

                # calculate risks and metrics
                risks, metrics = self.calculate_metrics_risks()

                # calculate metrics on target test labeled data
                scenario = f"{src_id}_to_{trg_id}"

                # append results
                table_results = table_results.append(pd.DataFrame([[scenario, run_id, *metrics]], columns=results_columns))
                table_risks = table_risks.append(pd.DataFrame([[scenario, run_id, *risks]], columns=risks_columns))

        average_metrics = [table_results[metric].mean() for metric in results_columns[2:]]
        std_metrics = [table_results[metric].std() for metric in results_columns[2:]]

        # add avg and std values
        table_results = table_results.append(pd.DataFrame([['mean', '-', *average_metrics]], columns=results_columns))
        table_results = table_results.append(pd.DataFrame([['std', '-', *std_metrics]], columns=results_columns))


        # log pandas dataframes to file or console
        print(table_results)
        print(table_risks)

        # save to file if needed
        table_results.to_csv(os.path.join(self.scenario_log_dir,"results.csv"))
        table_risks.to_csv(os.path.join(self.scenario_log_dir,"risks.csv"))




