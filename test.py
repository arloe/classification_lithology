import argparse
import torch
import os
import json
import itertools

from utils.util import ensure_dir

import numpy as np
import pandas as pd

from utils.util import get_instance
import model.model as module_arch

from data_loader.data_loaders import test_data

def main(  config
         , resume
         , data_filepath
         , lithology
         , indicator
         , outpath):
    
    """
    Load input and output feature names if they exist in the trained model configurations,
    otherwise manually feed them. 
    It is REQUIRED that the input features used for train and test procedures must be EXACTLY match in 
    their types, numbers, and even orders to obtain correct test results. 
    """
    # keys_in_config = config["data_loader"]["args"].keys()

    #    assert "in_features" in keys_in_config and "out_feature" in keys_in_config, \
    #        "Either in_features or out_feature not found in the loaded model"
    
    # import test dataset
    data_filepath = "./data/test_15_9-13.csv"
    in_features   = config["data_loader"]["args"]["in_features"]
    out_feature   = config["data_loader"]["args"]["out_feature"]
    lithology = "lithology.csv"
    
    depth_name = config["tensor_loader"]["args"]["depth_name"]
    max_depth = config["tensor_loader"]["args"]["max_depth"]
    
    chunk_depth = config["arch"]["args"]["chunk_depth"]
    batch_size = config["tensor_loader"]["args"]["batch_size"]
    shuffle = config["tensor_loader"]["args"]["shuffle"]
    
    # load scaler from checkpoint
    checkpoint = torch.load(resume)
    state_dict = checkpoint["state_dict"]
    scaler     = checkpoint["scaler"]
    
    loader, output_df = test_data(  data_filepath = data_filepath
                                  , in_features = in_features
                                  , out_feature = out_feature
                                  , depth_name = depth_name
                                  , max_depth = max_depth
                                  , chunk_depth = chunk_depth
                                  , batch_size = batch_size
                                  , shuffle = shuffle
                                  , scaler = scaler
                                  , lithology = lithology
                                  )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    model = get_instance(module_arch, "arch", config)

    # load weights from the trained model
    checkpoint = torch.load(resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model  = model.to(device)

    
    # predict test data
    pred_list = np.empty( output_df.shape[0] )
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate( loader ):
            data = data.to(device)
            output = model(data)
            
            predicted = torch.max(output, 1)[1].cpu().numpy()
            if batch_idx == 0:
                pred_list = predicted
                label_list = target.numpy()
            else:
                pred_list = np.concatenate( [pred_list, predicted] )
                label_list = np.concatenate( [label_list, target.numpy()] )

    output_df["label"] = label_list
    output_df["pred"]  = pred_list

    output_df.to_csv(os.path.join(outpath, "predict") + '_{:}.csv'.format(indicator), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Template")

    parser.add_argument("-r", "--resume", default=None, type = str, help = "path to the checkpoint (default: None)")
    parser.add_argument("-d", "--device", default=None, type = str, help = "indices of GPUs to enable (default: all)")
    parser.add_argument("-t", '--testfile', default=None, type = str, help = "path to test file(default: None)")
    parser.add_argument("-l", "--lithology", default = None, type = str, help = "file of lithology class")
    parser.add_argument("-o", "--outpath", default = None, type = str, help = "path output file saved")
    parser.add_argument("-i", "--indicator", default = None, type = str,
                        help = 'Time + indicator becomes the final directory name')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)["config"]
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # outpath = os.path.join("../output_file/classification", "" if args_outpath is None else args_outpath)
    outpath = os.path.join("./test_output", "" if args.outpath is None else args.outpath)
    ensure_dir(outpath)

    main(config, args.resume, args.testfile, args.lithology, args.indicator if args.indicator else "", outpath=outpath)
