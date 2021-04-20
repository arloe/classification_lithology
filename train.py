from __future__ import print_function

import os
import json
import argparse

import torch
            
import model.metric as module_metric
import model.model as module_arch

from utils import Logger
from utils.util import get_instance

from data_loader.data_loaders import load_data, data_to_tensor
from trainer.trainer import Trainer

def main(config, resume, indicator):
    # define parameters
    in_features = config["data_loader"]["args"]["in_features"]
    chunk_depth = config["arch"]["args"]["chunk_depth"]

    # Loading data
    train_raw_df, valid_raw_df = load_data( **config["data_loader"]["args"] )

    ## -- START: Training
    train_loader, train_df, scaler = data_to_tensor(  df = train_raw_df
                                                    , in_features = in_features
                                                    , chunk_depth = chunk_depth
                                                    , **config["tensor_loader"]["args"] )
    valid_loader, valid_df, _ = data_to_tensor(  df = valid_raw_df
                                               , in_features = in_features
                                               , chunk_depth = chunk_depth
                                               , scaler = scaler
                                               , **config["tensor_loader"]["args"] )
        
    # build model architecture
    model   = get_instance(module_arch, "arch", config)
    metrics = [ getattr(module_metric, met) for met in config["metrics"] ]
    trainable_params = filter(lambda p: p.requires_grad, model.parameters() )
    optimizer = get_instance(torch.optim, "optimizer", config, trainable_params)
    lr_scheduler = get_instance(torch.optim.lr_scheduler, "lr_scheduler_onplateau", config, optimizer)
    train_logger = Logger()
    
    # train
    trainer = Trainer(  model         = model
                      , metrics       = metrics
                      , optimizer     = optimizer
                      , indicator     = indicator
                      , chunk_depth   = chunk_depth
                      , resume        = resume
                      , config        = config
                      , train_loader  = train_loader
                      , valid_loader  = valid_loader
                      , lr_scheduler  = lr_scheduler
                      , train_logger  = train_logger
                      , scaler        = scaler
                      )
    trainer.train()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "PyTorch Template")
    parser.add_argument(  "-c", "--config", default = None, type = str
                        , help = "config file path (default: None)")
    parser.add_argument(  "-r", "--resume", default = None, type = str
                        , help = "path to latest checkpoint (default: None)")
    parser.add_argument(  "-d", "--device", default = None, type = str
                        , help = "indices of GPUs to enable (default: all)")
    parser.add_argument( "-i", "--indicator", default = None, type = str
                        , help = "Time + indicator becomes the final directory name")
    args = parser.parse_args()

    if args.config:
        # load config file
        with open(args.config) as handle:
            config = json.load( handle )
        # setting path to save trained models and log files
        path = os.path.join(config["trainer"]["save_dir"], config["name"])
    elif args.resume:
        # load config from checkpoint if new config file is not given.
        # Use '--config' and '--resume' together to fine-tune trained model with changed configurations.
        config = torch.load(args.resume)["config"]
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        
    main(config, args.resume, indicator = args.indicator if args.indicator else '')