import json
import argparse
import pandas as pd

from visdom import Visdom


parser = argparse.ArgumentParser(description='Visdom2CSV')

# Visdom parameters
parser.add_argument('--visdom-url', type=str, default="http://localhost",
                    help='visdom URL for graphs (default: http://localhost)')
parser.add_argument('--visdom-port', type=int, default="8097",
                    help='visdom port for graphs (default: 8097)')

# control params
parser.add_argument('--ls', type=str, default=None,
                    help='simply list the matching envs')

# feature params
parser.add_argument('--feature-name', type=str, default=None,
                    help='name of feature to grab metrics for')
parser.add_argument('--env-base-name', type=str, default=None,
                    help='base name for environment')

args = parser.parse_args()


def vis2dataframe(viz, env_base_name, feature_name):
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


def dataframes2csv(dfs):
    for k, v in dfs.items():
        v.to_csv(k + ".csv", index=False)


if __name__ == "__main__":
    # build the visdom object
    viz = Visdom(server=args.visdom_url, port=args.visdom_port, use_incoming_socket=False)

    # handle the --ls case
    if args.ls is not None:
        env_list = [e for e in viz.get_env_list() if args.ls in e]
        for env in env_list:
            print(env)

        exit(0)

    # grab the dataframes
    dfs = vis2dataframe(viz, args.env_base_name, args.feature_name)

    # write the dataframes to csv
    dataframes2csv(dfs)
