import numpy as np
from tqdm import tqdm
import time
# import os
# import pandas as pd

import torch
import torch.nn as nn
from base import BaseTrainer
# from model.loss import MeanPoweredErrorLoss, hub_loss

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model
                 , metrics, optimizer, resume, config
                 , train_loader, indicator
                 , valid_loader = None, valid_data_df = None, chunk_depth = None, lr_scheduler = None, train_logger = None
                 , class_weights = None
                 , scaler = None
                 ):
        # 190411 by hcho.
        # base_trainer requires loss as an input argument so I provide loss to None
        # as I don't wanna modifiy base_*.py files.
        super(Trainer, self).__init__(model, None, metrics, optimizer, resume, config, indicator, train_logger)
        self.config = config
        self.model_type = config["arch"]["type"]

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.do_validation = self.valid_loader is not None
        self.start_time = time.time()
        self.lr_scheduler = lr_scheduler
        self.best_acc = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.batch_size = self.config["tensor_loader"]["args"]["batch_size"]
        self.init_lr = self.optimizer.param_groups[0]["lr"]
        self.log_step = 100
        self.scaler = scaler

        
        if class_weights != None:
            class_weights = torch.Tensor( class_weights ).to(self.device)
            self.train_loss_metric = nn.NLLLoss(weight = class_weights)
            self.val_loss_metric   = nn.NLLLoss(weight = class_weights)
        else:
            self.train_loss_metric = nn.NLLLoss( )
            self.val_loss_metric   = nn.NLLLoss( )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}

            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.logger.info("\n\n**********************************************")
        self.logger.info("* Time elapsed: {:.1f} mins".format((time.time() - self.start_time) / 60))
        self.logger.info("* Batch size: {}".format(self.batch_size))
        self.logger.info("* Initial LR: {:.8f}".format(self.init_lr))
        self.logger.info("* Current LR: {:.8f}".format(self.optimizer.param_groups[0]['lr']))
        self.logger.info("* Aim epochs: {}".format(self.epochs))
        self.logger.info("********************************************")
        """Init cost value and set a descending learning rate"""
        
        agg_loss = 0.
        self.model.train()
        
        total_predicted, total_target = [], []
        with tqdm(total=len(self.train_loader)) as pbar:
            for batch_idx, (train_inputs, train_labels) in enumerate(self.train_loader):
                train_inputs = train_inputs.to(self.device)
                train_labels = train_labels.to(self.device)

                # batch normalization requires the number of data shuld be at least two.
                if len(train_inputs) <= 1:
                    pbar.update(1)
                    continue

                """The forward propagation"""
                # do a clean pass for unlabelled data
                outputs = self.model(train_inputs)  # return not used in training

                # calculate average supervised loss per input
                loss = self.train_loss_metric( outputs, train_labels.squeeze() )

                """back propagation"""
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                agg_loss += loss.data.item()

                """prob to class"""
                total_target += train_labels.squeeze().cpu().numpy().tolist()
                predicted = torch.max(outputs, 1)[1]
                total_predicted += predicted.cpu().numpy().tolist()

                pbar.set_description(('{}/{}'.format(batch_idx + 1, len(self.train_loader))))
                pbar.update(1)
               
        ##
        num_data = len(total_target)
        correct_predict = np.equal(total_predicted, total_target )
        train_acc = np.sum(correct_predict) / num_data
        
        num_batches = len(self.train_loader)
        log = {
            "train_loss": agg_loss / num_batches
        }
        self.writer.set_step(epoch)
        self.writer.add_scalar('total_loss', agg_loss / num_batches)

        if self.do_validation:
            val_log, val_loss, val_acc = self._valid_epoch(epoch, agg_loss / num_batches)
            log = {**log, **val_log}

        # decide wheter to update learning rate or not based on val loss
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_log['val_loss'])

        return log, agg_loss / num_batches, train_acc, val_loss, val_acc

    def _valid_epoch(self, epoch, train_sup_loss):
        self.model.eval()
        agg_supervised_loss = 0.
        total_target = []
        total_predicted = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.val_loss_metric( output, target.squeeze() )
                agg_supervised_loss += loss.item()
                
                total_target += target.cpu().numpy().tolist()
                predicted = torch.max(output, 1)[1]
                total_predicted += predicted.cpu().numpy().tolist()

                # TODO: Vp's index is now in constant of 6, but it could be different
                # if the input feature order changes in data_loader.py.
                # So be sure to improve the Vp indexing from constant to variable.


        num_data = len(total_target)
        correct_predict = np.equal(total_predicted, total_target )
        val_acc = np.sum(correct_predict) / num_data

        num_batches = len(self.valid_loader)
        val_loss = agg_supervised_loss / num_batches
        self.writer.set_step(epoch, 'valid')
        self.writer.add_scalar('val_loss', val_loss)
        
        # remember best result so far
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_acc = val_acc
            self.best_epoch = epoch
            
        self.model.train()

        self.logger.info('Epoch: {}\ttr_loss: {:.2f}\tvld_loss: {:.2f}\tvld_bst_loss: {:.2f}\tbest_acc: {:.2f} @ epoch {:d}'.format
                          ( epoch, train_sup_loss, val_loss, self.best_loss, self.best_acc, self.best_epoch ))
        return {'val_loss': val_loss}, val_loss, val_acc
