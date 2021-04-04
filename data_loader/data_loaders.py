import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

import torch
from torch.utils.data import TensorDataset, DataLoader


def load_data(  data_dir: str, input_filename: str
              , in_features: list, out_feature: str, well_name: str
              , validation_name: str, fold: str ):
    ""
    assert len(in_features) >= 1
    print("Loading Data....")
    
    # define path of dataset
    data_filename = os.path.join( data_dir, input_filename )
    
    # read dataset
    whole_data_df = pd.read_csv(filepath_or_buffer = data_filename)
    whole_data_df.columns = map(str.upper, whole_data_df.columns)
    
    # type of out_feature
    whole_data_df[out_feature] = whole_data_df[out_feature].astype("str")
    
    # remove NA
    validation_name = validation_name.upper()
    features = [well_name] + in_features + [out_feature] + [validation_name]
    df = whole_data_df.dropna( subset = features).loc[:, features ]
    
    # create label variable
    lith_value = df[ out_feature ].drop_duplicates().to_numpy()
    lith_value = sorted(lith_value)
    lith_class = np.linspace(start = 0, stop = len(lith_value) - 1, num = len(lith_value), dtype = "int" )
    label_df = pd.DataFrame( np.c_[lith_value, lith_class], columns = [out_feature, "label"] )
    
    df = pd.merge( left = df, right = label_df, how = "left", on = out_feature )
    df = df.set_index(keys = well_name)
    df["label"] = df["label"].astype("int")
    
    # train, validation data
    train_raw_df = df[ df[ validation_name ] != fold ]
    valid_raw_df = df[ df[ validation_name ] == fold ]

    return( train_raw_df, valid_raw_df )


def data_to_tensor(  df: pd.DataFrame, batch_size: int, shuffle: bool
                   , num_workers: int, max_depth: float
                   , in_features: list, depth_name: list
                   , chunk_depth: int, scaler = None ):
    
    assert len( in_features ) >= 1
    
    kwargs = {"num_workers": num_workers, "pin_memory": False}
    
    # define the name of feature
    in_features = list( map( str.upper, in_features) )

    # dataframe to array
    data_df = df.loc[:, in_features]
    label_orig = df.loc[:, "label" ].values
    data_orig  = df.loc[:, in_features].values
    
    # Calculate normalization
    if scaler == None: # train data
        scaler = RobustScaler(quantile_range=(1, 99)).fit( data_orig )
    else: pass         # validation data

    # Adopt normalization.
    # Note that different from other input features which are normalized by RobustScaler or StandadrdScaler,
    # depth feature is "linearly" normalized by diving the depths by a constant value.
    data_orig = scaler.transform( data_orig )
    if depth_name in data_df.columns:
        md_col_idx = data_df.columns.get_loc(depth_name)
        normalized_depth = data_orig[:, md_col_idx] / max_depth  # normalize depth in a different way
        data_orig[:, md_col_idx] = normalized_depth

    """ 
    Group multiple data points into a chunk(a.k.a window).
    To this end, first get the indices where each well begin and end in the train_data_df.
    Then every overlapping consecutive (with stride==1) data points with "chunk_depth" size are grouped together
    to form a one chunk. Here each chunk is considered as one input data.
    """
    # get begin and end indices of cl
    begin_idx, end_idx = list(), list()
    for well in df.index.drop_duplicates():
        indices = np.argwhere( df.index == well )
        begin_idx.append( indices.min())
        end_idx.append( indices.max())

    assert len(begin_idx) == len(end_idx)

    # data point grouping for training
    data, label, idx = [], [], []
    for begin, end in zip( begin_idx, end_idx):
        while begin + chunk_depth - 1 <= end:  # for every consecutive chunk_depth points
            idx_ = int(np.floor( np.median( [begin, begin + chunk_depth] ) )) # index of center
            data.append( data_orig[begin : begin + chunk_depth] )
            label.append( label_orig[idx_] )
            idx.append( idx_ )
            begin += 1
            
    data  = np.asarray( data )
    label = np.asarray( label )
    idx   = np.asarray( idx )

    # dataset
    output_df = df.reset_index().loc[ idx, ]
    
    # for data, transform N x H x W  to N x 1 x H x W format.
    data = torch.unsqueeze(torch.FloatTensor( data ), 1)
    
    # for label, transform N x H x 1 to N x H format.
    label = torch.squeeze(torch.LongTensor(label))
    
    dataset = TensorDataset( data, label )
    loader  = DataLoader( dataset, batch_size = batch_size, shuffle = shuffle, **kwargs )

    return loader, output_df, scaler
