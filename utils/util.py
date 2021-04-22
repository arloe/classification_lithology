import os
import numpy as np
import matplotlib.pyplot as plt
import time


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def draw_plot(path, writer, epoch, total_target, total_predicted, out_feature):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # draw plot along with x-axis
    ax.plot(total_target, c='b', linewidth=0.5, label='target ' + out_feature)
    ax.plot(total_predicted, c='r', linewidth=0.5, label='predicted ' + out_feature, alpha=0.7)
    plt.xlabel('depth')
    plt.ylabel(out_feature)

    _ = plt.legend()
    plt.savefig(path + '-epochs{:03d}.jpg'.format(epoch), bbox_inches='tight', format='jpg', dpi=1000)
    if writer is not None:
        writer.add_figure("matplotlib", fig)
    plt.clf()
    time.sleep(0.1)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sort_index = sorted(range(len(total_target)), key=total_target.__getitem__)
    ax.scatter(np.asarray(total_target)[sort_index], np.asarray(total_predicted)[sort_index], s=0.5, alpha=0.5)
    plt.xlabel('target ' + out_feature)
    plt.ylabel('predicted ' + out_feature)
    plt.grid()
    plt.savefig(path + '-epochs{:03d}-mcman_curve.jpg'.format(epoch), bbox_inches='tight', format='jpg', dpi=1000)
    # take a sleep to prevent hanging
    time.sleep(0.1)
    plt.close(fig)

def get_instance(module, name, config, *args):
    return getattr(module, config[name]["type"])(*args, **config[name]["args"])