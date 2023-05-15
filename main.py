from trainers.train import Trainer
from trainers.test import Test

import argparse
parser = argparse.ArgumentParser()

if __name__ == "__main__":

    # ========  Experiments Phase ================
    parser.add_argument('--phase',               default='test',         type=str, help='train, test')

    # ========  Experiments Name ================
    parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
    parser.add_argument('--exp_name',               default='EXP1',         type=str, help='experiment name')

    # ========= Select the DA methods ============
    parser.add_argument('--da_method',              default='Deep_Coral',               type=str, help='DANN, Deep_Coral, WDGRL, MMDA, VADA, DIRT, CDAN, ADDA, HoMM, CoDATS')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path',              default=r'../ADATIME_data',                  type=str, help='Path containing datase2t')
    parser.add_argument('--dataset',                default='HAR',                      type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone',               default='CNN',                      type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs',               default=1,                          type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device',                 default= "cuda",                   type=str, help='cpu or cuda')

    args = parser.parse_args()

    if args.phase == 'train':
        trainer = Trainer(args)
        trainer.train()
    elif args.phase == 'test':
        tester = Test(args)
        tester.test()



#TODO:
# 1- Change the naming of the functions ---> ( Done)
# 2- Change the algorithms following DCORAL 
# 4- Add pretrain based methods (ADDA, MCD)
# 5- Add the best hparams 
