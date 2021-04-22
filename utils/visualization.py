from torch.utils.tensorboard import SummaryWriter

class WriterTensorboardX():
    def __init__(self, writer_dir, logger, enable):
        self.writer = None
        if enable:
            log_path = writer_dir
            try:
                # self.writer = importlib.import_module('tensorboard').SummaryWriter(log_path)
                self.writer = SummaryWriter(log_path)
            except ImportError:
                message = "Warning: TensorboardX visualization is configured to use, but currently not installed on " \
                    "this machine. Please install the package by 'pip install tensorboardx' command or turn " \
                    "off the option in the 'config.json' file."
                logger.warning(message)
        self.step = 0
        self.mode = ''

        self.tb_writer_ftns = [
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding', 'add_figure'
        ]
        self.tag_mode_exceptions = ['add_histogram', 'add_embedding']

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add mode(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(self.mode, tag)
                    add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object 'WriterTensorboardX' has no attribute '{}'".format(name))
            return attr


# =============================================================================
# -- 201013 by Choi
# -- Add Visualization(MC Kim)
# -- history plot & confusion matrix & graph ...
# =============================================================================

import os
import re
import torch

from matplotlib import pyplot as plt

import numpy as np


class history():
    def __init__(self, writer_dir: str, loss_str: str = "loss", eval_str: str = "accuracy"):
        self.writer_dir = writer_dir
        self.loss_str   = loss_str
        self.eval_str   = eval_str

    def single_graph(self, epoch, train: list, valid: list, label: str, title: str, silence: bool = False):
        try:
            plt.plot( epoch, train, "bo", label = "Training " + label, ms = .5 )
        except:
            pass
        try:
            plt.plot( epoch, valid, "r", label = "Validation " + label, linewidth = 1. )
        except:
            pass
        if silence != False:
            plt.title("Training and Validation " + title)
        plt.xlabel("Epochs")
        plt.ylabel(label)    
        plt.legend()
    
    def plot(self, train_loss: list, train_acc: list, valid_loss: list, valid_acc: list):
        assert len( train_loss ) == len( valid_loss )
        assert len( train_acc )  == len( valid_acc )
        
        # define
        epoch = range( 1, len( train_loss ) + 1 )
        loss_str = self.loss_str
        eval_str = self.eval_str

        # loss curve
        fig = plt.figure(figsize = (8, 6))
        plt.subplot(221)
        self.single_graph( epoch = epoch, train = train_loss, valid = valid_loss, label = loss_str, title = "LOSS", silence = True )
        plt.subplot(223)
        self.single_graph( epoch = epoch, train = train_loss, valid = valid_loss, label = loss_str, title = "LOSS", silence = False )
        y_max = max( np.max( valid_loss[int(len(valid_loss)/2):]), np.max(train_loss[int(len(train_loss)/2):] ) ) * 1.1
        plt.ylim([0, y_max])

        # accuracy curve
        plt.subplot(224)
        self.single_graph( epoch = epoch, train = train_acc, valid = valid_acc, label = eval_str, title = "ACCURACY", silence = True )
        plt.subplot(222)
        self.single_graph( epoch = epoch, train = train_acc, valid = valid_acc, label = eval_str, title = "ACCURACY", silence = False )
        y_max = max( np.max( valid_acc[int(len(valid_acc)/2):]), np.max(train_acc[int(len(train_acc)/2):] ) ) * 1.1
        plt.ylim([0, y_max])
        
        # option
        plt.suptitle("LOSS CURVE")
        plt.tight_layout()
        plt.subplots_adjust(top = 0.85)
        
        # save
        plt.savefig( os.path.join( self.writer_dir, "history.png" ) )
        plt.close( fig )

def _predict(model, dataset, valid = True, path = None, best_index = None):
    """
    """
    if valid:
        filenames = os.listdir( path = path )
        if "best_model.pth" in filenames:
            filename = "best_model.pth"
        else:
            epoch = sum( [ re.findall( pattern = "\d+", string = element ) for element in filenames ], [] )
            epoch = [ int(element) for element in epoch ]
            
            best_epoch = min( epoch, key = lambda x: abs(x - best_index) )
            filename   = "checkpoint-epoch" + str(best_epoch) + ".pth"
        
        checkpoint = torch.load( os.path.join(path, filename) )
        state_dict = checkpoint["state_dict"]
        model.load_state_dict( state_dict )
    else:
        model = model
        
    # predict
    model.eval()
    pred_prob  = model( dataset )
    pred_prob  = torch.exp( pred_prob )
    pred_label = torch.max( pred_prob, dim = 1 )[1]
    
    pred_prob  = pred_prob.to("cpu").detach().numpy()
    pred_label = pred_label.to("cpu").detach().numpy()
    
    return pred_prob, pred_label

    
# =============================================================================
# -- visualization (Log plot)
# =============================================================================
def visualization( df, features, depth_name ):
    lithology_numbers = {  0: {"lith": "Sandstone", "hatch": "..", "color": "#ffff00"}
                     , 1: {"lith": "Shale", "hatch": "--", "color": "#bebebe"}
                     , 2: {"lith": "Sandstone/Shale", "hatch": "-.", "color": "#ffe119"}
                     , 3: {"lith": "Limestone", "hatch": "+", "color": "#80ffff"}
                     , 4: {"lith": "Chalk", "hatch": "..", "color": "#80ffff"}
                     , 5: {"lith": "Dolomite", "hatch": "-/", "color": "#8080ff"}
                     , 6: {"lith": "Marl", "hatch": "", "color": "#7cfc00"}
                     , 7: {"lith": "Halite", "hatch": "x", "color": "#7ddfbe"}
                     , 8: {"lith": "Coal", "hatch": "", "color": "black"}
                     , 9: {"lith": "Tuff", "hatch": "||", "color": "#ff8c00"} 
                     }
    
    fig, ax = plt.subplots( figsize = (10, 15) )

    n = len(features)

    for i in range(n):
        ax = plt.subplot2grid( (2, 2*n + 6), (0, 2*i), rowspan = 1, colspan = 2)
        ax.plot( features[i], depth_name, data = df, linewidth = .5, c = "C"+str(i) )
        plt.fill_betweenx( depth_name, features[i], data = df, facecolor = "C"+str(i), alpha = .3)
        # set x, y label
        ax.set_xlabel( features[i] )
        if i == 0:
            ax.set_ylabel( depth_name )
        else:
            ax.set_ylabel( "" )
            plt.setp( ax.get_yticklabels(), visible = False )

        ax.grid( which = "major", color = "lightgrey", linestyle = "--" )
        ax.xaxis.set_ticks_position( "top" )
        ax.xaxis.set_label_position( "top" )

    ax = plt.subplot2grid( (2, 2*n + 6), (0, 2*n + 1), rowspan = 1, colspan = 2)
    ax.set_xlabel("Lithology")
    ax.set_xlim(0, 1)
    plt.setp( ax.get_yticklabels(), visible = False )

    for key in lithology_numbers.keys():
        color = lithology_numbers[key]["color"]
        hatch = lithology_numbers[key]["color"]
        ax.fill_betweenx( depth_name, 0, "label"
                         , data = df
                         , where = (df["label"] >= key),
                         facecolor = color, hatch = hatch)
    ax.xaxis.set_ticks_position( "top" )
    ax.xaxis.set_label_position( "top" )

    ax = plt.subplot2grid( (2, 2*n + 6), (0, 2*n + 4), rowspan = 1, colspan = 2)
    ax.set_xlabel("Predicted")
    ax.set_xlim(0, 1)
    plt.setp( ax.get_yticklabels(), visible = False )

    for key in lithology_numbers.keys():
        color = lithology_numbers[key]["color"]
        hatch = lithology_numbers[key]["color"]
        ax.fill_betweenx(  df[depth_name], 0, df["pred"]
                         , where = (df["label"]>=key)
                         , facecolor = color
                         , hatch = hatch
                        )
    ax.xaxis.set_ticks_position( "top" )
    ax.xaxis.set_label_position( "top" )
    
    plt.show()

# =============================================================================
# -- confusion matrix
# =============================================================================
import itertools
from sklearn.metrics import confusion_matrix

def confusion_matrix_visualization( df ):
    cm = confusion_matrix( df["label"], df["pred"])
    classes = df["label"].drop_duplicates().to_list()
    classes = list(set(classes))

    plt.imshow(cm, interpolation = "nearest", cmap = plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")
    plt.title( "Confusion Matrix" )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()