import torch
import torch.nn as nn
import numpy as np
import itertools    

from models.models import classifier, ReverseLayerF, Discriminator, RandomLayer, Discriminator_CDAN, \
    codats_classifier, AdvSKM_Disc, CNN_ATTN
from models.loss import MMD_loss, CORAL, ConditionalEntropyLoss, VAT, LMMD_loss, HoMM_loss, NTXentLoss, SupConLoss
from utils import EMA
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import torch.nn. functional as F

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs, backbone):
        super(Algorithm, self).__init__()
        self.configs = configs

        self.cross_entropy = nn.CrossEntropyLoss()
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)


    # update function is common to all algorithms
    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            
            # training loop 
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())


            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        
        last_model = self.network.state_dict()

        return last_model, best_model
    
    # train loop vary from one method to another
    def training_epoch(self, *args, **kwargs):
        raise NotImplementedError
       

class NO_ADAPT(Algorithm):
    """
    Lower bound: train on source and test on target.
    """
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):
        for src_x, src_y in src_loader:
            
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()
    

class TARGET_ONLY(Algorithm):
    """
    Upper bound: train on target and test on target.
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        for trg_x, trg_y in trg_loader:

            trg_x, trg_y = trg_x.to(self.device), trg_y.to(self.device)

            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            trg_cls_loss = self.cross_entropy(trg_pred, trg_y)

            loss = trg_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Trg_cls_loss': trg_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


class Deep_Coral(Algorithm):
    """
    Deep Coral: https://arxiv.org/abs/1607.01719
    """
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # correlation alignment loss
        self.coral = CORAL()


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        # add if statement

        if len(src_loader) > len(trg_loader):
            joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        else:
            joint_loader =enumerate(zip(itertools.cycle(src_loader), trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            trg_feat = self.feature_extractor(trg_x)

            coral_loss = self.coral(src_feat, trg_feat)

            loss = self.hparams["coral_wt"] * coral_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses = {'Total_loss': loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                    'coral_loss': coral_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class MMDA(Algorithm):
    """
    MMDA: https://arxiv.org/abs/1901.00282
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd = MMD_loss()
        self.coral = CORAL()
        self.cond_ent = ConditionalEntropyLoss()


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            trg_feat = self.feature_extractor(trg_x)
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            trg_feat = self.feature_extractor(trg_x)

            coral_loss = self.coral(src_feat, trg_feat)
            mmd_loss = self.mmd(src_feat, trg_feat)
            cond_ent_loss = self.cond_ent(trg_feat)

            loss = self.hparams["coral_wt"] * coral_loss + \
                self.hparams["mmd_wt"] * mmd_loss + \
                self.hparams["cond_ent_wt"] * cond_ent_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'Coral_loss': coral_loss.item(), 'MMD_loss': mmd_loss.item(),
                    'cond_ent_wt': cond_ent_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
            
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


class DANN(Algorithm):
    """
    DANN: https://arxiv.org/abs/1505.07818
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        
        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Domain Discriminator
        self.domain_classifier = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):
        # Combine dataloaders
        # Method 1 (min len of both domains)
        # joint_loader = enumerate(zip(src_loader, trg_loader))

        # Method 2 (max len of both domains)
        # joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:

            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            
            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # Domain classification loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_classifier(trg_feat_reversed)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
           
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class CDAN(Algorithm):
    """
    CDAN: https://arxiv.org/abs/1705.10667
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)


        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment Losses
        self.criterion_cond = ConditionalEntropyLoss().to(device)

        self.domain_classifier = Discriminator_CDAN(configs)
        self.random_layer = RandomLayer([configs.features_len * configs.final_out_channels, configs.num_classes],
                                        configs.features_len * configs.final_out_channels)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"])

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            # prepare true domain labels
            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

            # source features and predictions
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # target features and predictions
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # concatenate features and predictions
            feat_concat = torch.cat((src_feat, trg_feat), dim=0)
            pred_concat = torch.cat((src_pred, trg_pred), dim=0)

            # Domain classification loss
            feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1)).detach()
            disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
            disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # update Domain classification
            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            self.optimizer_disc.step()

            # prepare fake domain labels for training the feature extractor
            domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
            domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

            # Repeat predictions after updating discriminator
            feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1))
            disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
            # loss of domain discriminator according to fake labels

            domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # conditional entropy loss.
            loss_trg_cent = self.criterion_cond(trg_pred)

            # total loss
            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["cond_ent_wt"] * loss_trg_cent

            # update feature extractor
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                    'cond_ent_loss': loss_trg_cent.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)
        self.lr_scheduler.step()

class DIRT(Algorithm):
    """
    DIRT-T: https://arxiv.org/abs/1802.08735
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


        # Aligment losses
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.vat_loss = VAT(self.network, device).to(device)
        self.ema = EMA(0.998)
        self.ema.register(self.network)

        # Discriminator
        self.domain_classifier = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
       
    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            # prepare true domain labels
            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # target features and predictions
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # concatenate features and predictions
            feat_concat = torch.cat((src_feat, trg_feat), dim=0)

            # Domain classification loss
            disc_prediction = self.domain_classifier(feat_concat.detach())
            disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # update Domain classification
            self.optimizer_disc.zero_grad()
            disc_loss.backward()
            self.optimizer_disc.step()

            # prepare fake domain labels for training the feature extractor
            domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
            domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
            domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

            # Repeat predictions after updating discriminator
            disc_prediction = self.domain_classifier(feat_concat)

            # loss of domain discriminator according to fake labels
            domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # conditional entropy loss.
            loss_trg_cent = self.criterion_cond(trg_pred)

            # Virual advariarial training loss
            loss_src_vat = self.vat_loss(src_x, src_pred)
            loss_trg_vat = self.vat_loss(trg_x, trg_pred)
            total_vat = loss_src_vat + loss_trg_vat
            # total loss
            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["cond_ent_wt"] * loss_trg_cent + self.hparams["vat_loss_wt"] * total_vat

            # update exponential moving average
            self.ema(self.network)

            # update feature extractor
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                    'cond_ent_loss': loss_trg_cent.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class DSAN(Algorithm):
    """
    DSAN: https://ieeexplore.ieee.org/document/9085896
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Alignment losses
        self.loss_LMMD = LMMD_loss(device=device, class_num=configs.num_classes).to(device)

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)        # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # calculate lmmd loss
            domain_loss = self.loss_LMMD.get_loss(src_feat, trg_feat, src_y, torch.nn.functional.softmax(trg_pred, dim=1))

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'LMMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class HoMM(Algorithm):
    """
    HoMM: https://arxiv.org/pdf/1912.11976.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # aligment losses
        self.coral = CORAL()
        self.HoMM_loss = HoMM_loss()

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
            
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # calculate lmmd loss
            domain_loss = self.HoMM_loss(src_feat, trg_feat)

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'HoMM_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
            
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


class DDC(Algorithm):
    """
    DDC: https://arxiv.org/abs/1412.3474
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd_loss = MMD_loss()

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
            # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # calculate mmd loss
            domain_loss = self.mmd_loss(src_feat, trg_feat)

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * domain_loss + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'MMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class CoDATS(Algorithm):
    """
    CoDATS: https://arxiv.org/pdf/2005.10996.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # we replace the original classifier with codats the classifier
        # remember to use same name of self.classifier, as we use it for the model evaluation
        self.classifier = codats_classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


        # Domain classifier
        self.domain_classifier = Discriminator(configs)

        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
        
            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            trg_feat = self.feature_extractor(trg_x)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # Domain classification loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_classifier(trg_feat_reversed)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses =  {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class AdvSKM(Algorithm):
    """
    AdvSKM: https://www.ijcai.org/proceedings/2021/0378.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd_loss = MMD_loss()
        self.AdvSKM_embedder = AdvSKM_Disc(configs).to(device)
        self.optimizer_disc = torch.optim.Adam(
            self.AdvSKM_embedder.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features
            
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)

            source_embedding_disc = self.AdvSKM_embedder(src_feat.detach())
            target_embedding_disc = self.AdvSKM_embedder(trg_feat.detach())
            mmd_loss = - self.mmd_loss(source_embedding_disc, target_embedding_disc)
            mmd_loss.requires_grad = True

            # update discriminator
            self.optimizer_disc.zero_grad()
            mmd_loss.backward()
            self.optimizer_disc.step()

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # domain loss.
            source_embedding_disc = self.AdvSKM_embedder(src_feat)
            target_embedding_disc = self.AdvSKM_embedder(trg_feat)

            mmd_loss_adv = self.mmd_loss(source_embedding_disc, target_embedding_disc)
            mmd_loss_adv.requires_grad = True

            # calculate the total loss
            loss = self.hparams["domain_loss_wt"] * mmd_loss_adv + \
                self.hparams["src_cls_loss_wt"] * src_cls_loss

            # update optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(), 'MMD_loss': mmd_loss_adv.item(), 'Src_cls_loss': src_cls_loss.item()}
            for key, val in losses.items():
                    avg_meter[key].update(val, 32)

        self.lr_scheduler.step()

class SASA(Algorithm):
    
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # feature_length for classifier
        configs.features_len = 1
        self.classifier = classifier(configs)
        # feature length for feature extractor
        configs.features_len = 1
        self.feature_extractor = CNN_ATTN(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features

            # Extract features
            src_feature = self.feature_extractor(src_x)
            tgt_feature = self.feature_extractor(trg_x)

            # source classification loss
            y_pred = self.classifier(src_feature)
            src_cls_loss = self.cross_entropy(y_pred, src_y)

            # MMD loss
            domain_loss_intra = self.mmd_loss(src_struct=src_feature,
                                            tgt_struct=tgt_feature, weight=self.hparams['domain_loss_wt'])

            # total loss
            total_loss = self.hparams['src_cls_loss_wt'] * src_cls_loss + domain_loss_intra

            # remove old gradients
            self.optimizer.zero_grad()
            # calculate gradients
            total_loss.backward()
            # update the weights
            self.optimizer.step()

            losses =  {'Total_loss': total_loss.item(), 'MMD_loss': domain_loss_intra.item(),
                    'Src_cls_loss': src_cls_loss.item()}
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()
    def mmd_loss(self, src_struct, tgt_struct, weight):
        delta = torch.mean(src_struct - tgt_struct, dim=-2)
        loss_value = torch.norm(delta, 2) * weight
        return loss_value


class CoTMix(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

         # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.contrastive_loss = NTXentLoss(device, hparams["batch_size"], 0.2, True)
        self.entropy_loss = ConditionalEntropyLoss()
        self.sup_contrastive_loss = SupConLoss(device)

    def training_epoch(self,src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)         # extract source features

            # ====== Temporal Mixup =====================
            src_dominant, trg_dominant = self.temporal_mixup(src_x, trg_x)

            # ====== Source =====================
            self.optimizer.zero_grad()

            # Src original features
            src_orig_feat = self.feature_extractor(src_x)
            src_orig_logits = self.classifier(src_orig_feat)

            # Target original features
            trg_orig_feat = self.feature_extractor(trg_x)
            trg_orig_logits = self.classifier(trg_orig_feat)

            # -----------  The two main losses
            # Cross-Entropy loss
            src_cls_loss = self.cross_entropy(src_orig_logits, src_y)
            loss = src_cls_loss * round(self.hparams["src_cls_weight"], 2)

            # Target Entropy loss
            trg_entropy_loss = self.entropy_loss(trg_orig_logits)
            loss += trg_entropy_loss * round(self.hparams["trg_entropy_weight"], 2)

            # -----------  Auxiliary losses
            # Extract source-dominant mixup features.
            src_dominant_feat = self.feature_extractor(src_dominant)
            src_dominant_logits = self.classifier(src_dominant_feat)

            # supervised contrastive loss on source domain side
            src_concat = torch.cat([src_orig_logits.unsqueeze(1), src_dominant_logits.unsqueeze(1)], dim=1)
            src_supcon_loss = self.sup_contrastive_loss(src_concat, src_y)
            loss += src_supcon_loss * round(self.hparams["src_supCon_weight"], 2)

            # Extract target-dominant mixup features.
            trg_dominant_feat = self.feature_extractor(trg_dominant)
            trg_dominant_logits = self.classifier(trg_dominant_feat)

            # Unsupervised contrastive loss on target domain side
            trg_con_loss = self.contrastive_loss(trg_orig_logits, trg_dominant_logits)
            loss += trg_con_loss * round(self.hparams["trg_cont_weight"], 2)

            loss.backward()
            self.optimizer.step()

            losses =  {'Total_loss': loss.item(),
                    'src_cls_loss': src_cls_loss.item(),
                    'trg_entropy_loss': trg_entropy_loss.item(),
                    'src_supcon_loss': src_supcon_loss.item(),
                    'trg_con_loss': trg_con_loss.item()
                    }
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()           

    def temporal_mixup(self,src_x, trg_x):
        
        mix_ratio = round(self.hparams["mix_ratio"], 2)
        temporal_shift = self.hparams["temporal_shift"]
        h = temporal_shift // 2  # half

        src_dominant = mix_ratio * src_x + (1 - mix_ratio) * \
                    torch.mean(torch.stack([torch.roll(trg_x, -i, 2) for i in range(-h, h)], 2), 2)

        trg_dominant = mix_ratio * trg_x + (1 - mix_ratio) * \
                    torch.mean(torch.stack([torch.roll(src_x, -i, 2) for i in range(-h, h)], 2), 2)
        
        return src_dominant, trg_dominant
    


# Untied Approaches: (MCD)
class MCD(Algorithm):
    """
    Maximum Classifier Discrepancy for Unsupervised Domain Adaptation
    MCD: https://arxiv.org/pdf/1712.02560.pdf
    """

    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.classifier2 = classifier(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)


        # optimizer and scheduler
        self.optimizer_fe = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
                # optimizer and scheduler
        self.optimizer_c1 = torch.optim.Adam(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
                # optimizer and scheduler
        self.optimizer_c2 = torch.optim.Adam(
            self.classifier2.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.lr_scheduler_fe = StepLR(self.optimizer_fe, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler_c1 = StepLR(self.optimizer_c1, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler_c2 = StepLR(self.optimizer_c2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Aligment losses
        self.mmd_loss = MMD_loss()

    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            
            # source pretraining loop 
            self.pretrain_epoch(src_loader, avg_meter)

            # training loop 
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())


            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        
        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader,avg_meter):
        for src_x, src_y in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)
          
            src_feat = self.feature_extractor(src_x)
            src_pred1 = self.classifier(src_feat)
            src_pred2 = self.classifier2(src_feat)

            src_cls_loss1 = self.cross_entropy(src_pred1, src_y)
            src_cls_loss2 = self.cross_entropy(src_pred2, src_y)

            loss = src_cls_loss1 + src_cls_loss2

            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()
            self.optimizer_fe.zero_grad()

            loss.backward()

            self.optimizer_c1.step()
            self.optimizer_c2.step()
            self.optimizer_fe.step()

            
            losses = {'Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)           # extract source features
            

            # extract source features
            src_feat = self.feature_extractor(src_x)
            src_pred1 = self.classifier(src_feat)
            src_pred2 = self.classifier2(src_feat)

            # source losses
            src_cls_loss1 = self.cross_entropy(src_pred1, src_y)
            src_cls_loss2 = self.cross_entropy(src_pred2, src_y)
            loss_s = src_cls_loss1 + src_cls_loss2
            

            # Freeze the feature extractor
            for k, v in self.feature_extractor.named_parameters():
                v.requires_grad = False
            # update C1 and C2 to maximize their difference on target sample
            trg_feat = self.feature_extractor(trg_x) 
            trg_pred1 = self.classifier(trg_feat.detach())
            trg_pred2 = self.classifier2(trg_feat.detach())


            loss_dis = self.discrepancy(trg_pred1, trg_pred2)

            loss = loss_s - loss_dis
            
            loss.backward()
            self.optimizer_c1.step()
            self.optimizer_c2.step()

            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()
            self.optimizer_fe.zero_grad()

            # Freeze the classifiers
            for k, v in self.classifier.named_parameters():
                v.requires_grad = False
            for k, v in self.classifier2.named_parameters():
                v.requires_grad = False
                        # Freeze the feature extractor
            for k, v in self.feature_extractor.named_parameters():
                v.requires_grad = True
            # update feature extractor to minimize the discrepaqncy on target samples
            trg_feat = self.feature_extractor(trg_x)        
            trg_pred1 = self.classifier(trg_feat)
            trg_pred2 = self.classifier2(trg_feat)


            loss_dis_t = self.discrepancy(trg_pred1, trg_pred2)
            domain_loss = self.hparams["domain_loss_wt"] * loss_dis_t 

            domain_loss.backward()
            self.optimizer_fe.step()

            self.optimizer_fe.zero_grad()
            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()


            losses =  {'Total_loss': loss.item(), 'MMD_loss': domain_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler_fe.step()
        self.lr_scheduler_c1.step()
        self.lr_scheduler_c2.step()

    def discrepancy(self, out1, out2):

        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))
