import os
import json
import argparse
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})
# plt.rcParams.update({'font.size': 5})
plt.rcParams.update({'savefig.dpi': 1000})
# set global seaborn plot stuff
# sns.set_context("paper", rc={"font.size":15,"axes.titlesize":15,"axes.labelsize":5})

from visdom import Visdom


parser = argparse.ArgumentParser(description='Visdom2PNG')

# Visdom parameters
parser.add_argument('--visdom-url', type=str, default="http://localhost",
                    help='visdom URL for graphs (default: http://localhost)')
parser.add_argument('--visdom-port', type=int, default="8097",
                    help='visdom port for graphs (default: 8097)')

# feature params
parser.add_argument('--feature-name', type=str, required=True,
                    help='(Required) Name of feature to grab metrics for')
parser.add_argument('--env-base-names', nargs='+', required=True,
                    help='(Required) List of base environment names')

# plot params
parser.add_argument('--title', type=str, default="",
                    help='title for plot')
parser.add_argument('--x-label', type=str, default="epoch",
                    help='x-label for plot (default: epoch)')
parser.add_argument('--y-label', type=str, default="test-accuracy",
                    help='y-label for plot (default: test-accuracy)')
parser.add_argument('--legends', nargs='+', required=False, default=None,
                    help='(optional) List of legend overrides, none uses env_name')
parser.add_argument('--output', type=str, default="out.png",
                    help='output image filename (default: out.png)')

args = parser.parse_args()


def vis2dataframe(viz, env_base_name, feature_name):
    """ Helper to return a dict with env_base_name_* : pd.Dataframe

    :param viz: visdom instance
    :param env_base_name: the base environment name, grabs all of these
    :param feature_name: the feature to pull from the environment
    :returns: a dict of many
    :rtype:

    """

    assert env_base_name is not None, "need env base name"
    assert feature_name is not None, "need feature name"
    envs = [e for e in viz.get_env_list() if env_base_name in e]
    assert len(envs) > 0, "no matching envs detected"

    # parse all envs
    json_blobs = [json.loads(viz.get_window_data(win=None, env=e)) for e in envs]
    print("found {} matching envs for {}".format(len(json_blobs), env_base_name))

    dfs = {}
    for env_name, env in zip(envs, json_blobs): # iterate over all different environments
        for window, value in env.items():       # over all windows in that env
            if 'title' in list(value.keys()):
                if value['title'] == feature_name:
                    assert len(value['content']['data']) == 1, "content has multiple items, how to handle?"
                    x = value['content']['data'][0]['x']
                    y = value['content']['data'][0]['y']
                    dfs[env_name] = pd.DataFrame({'x': x, 'y': y})

    return dfs


def read_x_y(dfs, maxval=-1):
    """reads files in directory and produce x and y

    :param dfs: the map of dataframes
    :param maxval: the index of the maximum number of samples to read
    :returns: dataframe with x and y
    :rtype: pd.Dataframe

    """
    csvs = [df.values for df in list(dfs.values())]
    x = np.hstack([ds[:, 0][0:maxval] for ds in csvs])
    y = np.hstack([ds[:, 1][0:maxval] for ds in csvs])
    return pd.DataFrame({args.x_label: x, args.y_label: y})


def read_x_y_multisample(dfs, maxval=-1):
    """reads files in directory and produce x and y, but stacks in different order

    :param dfs: the map of dataframes
    :param maxval: the index of the maximum number of samples to read
    :returns: dataframe with x and y
    :rtype: pd.Dataframe

    """
    csvs = [df.values for df in list(dfs.values())]
    x = [np.expand_dims(ds[:, 0][0:maxval], -1) for ds in csvs]
    y = [np.expand_dims(ds[:, 1][0:maxval], -1) for ds in csvs]
    return pd.DataFrame({args.x_label: x, args.y_label: y})


def print_max_min_mean_std(df, name, key='test-accuracy'):
    """ Print the max, min, mean and 1-std of the dataframe for min and max

    :param df: the pandas dataframe
    :param name: name to prefix
    :param key: the key to use for computation
    :returns: None
    :rtype: None

    """
    maximums = [np.max(run) for run in df[key].values]
    print('{} [max] : \t\tmax {} |  {} +/- {}'.format(name,
                                                      np.max(maximums),
                                                      np.mean(maximums),
                                                      np.std(maximums)))

    minimums = [np.min(run) for run in df[key].values]
    print('{} [min] : \t\tmin {} |  {} +/- {}'.format(name,
                                                      np.min(minimums),
                                                      np.mean(minimums),
                                                      np.std(minimums)))


def dump_stats(dfs):
    """ print statistics for all the dfs

    :param dfs: the list of dataframes
    :returns: None
    :rtype: None

    """
    for df, name in zip(dfs, args.env_base_names):
        concat_ms_df = read_x_y_multisample(df, maxval=-1)
        print_max_min_mean_std(concat_ms_df, name=name, key=args.y_label)


if __name__ == "__main__":
    # build the visdom object
    viz = Visdom(server=args.visdom_url, port=args.visdom_port, use_incoming_socket=False)

    # grab 1 dict per env_base name
    # each dict can have many key:values for the different instances of the same env
    dfs = [vis2dataframe(viz, env_base_name_i, args.feature_name)
           for env_base_name_i in args.env_base_names]
    assert len(dfs) == len(args.env_base_names)

    # print some stats
    dump_stats(dfs)

    # for each dict from above, merge them together and plot it
    plt.figure()
    for i, (df, name) in enumerate(zip(dfs, args.env_base_names)):
        merged_df = read_x_y(df, maxval=-1)
        legend_name = name if args.legends is None else args.legends[i]
        sns.lineplot(x=args.x_label, y=args.y_label, data=merged_df, label=legend_name)

    plt.title(args.title)
    plt.savefig(args.output, bbox_inches='tight')
