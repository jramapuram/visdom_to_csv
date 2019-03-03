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

# output storing
parser.add_argument('--output', type=str, default="out.csv",
                    help='name of output file')

args = parser.parse_args()


def recurse_and_find(m, key):
    found = None
    for k,v in m.items():
        if isinstance(v, dict):
            found = recurse_and_find(v, key)
            if found is not None:
                return found

        if k == 'title':
            print("title v = ", v)

        if k == key:
            return v


if __name__ == "__main__":
    # build the visdom object
    viz = Visdom(server=args.visdom_url, port=args.visdom_port, use_incoming_socket=False)

    # handle the --ls case
    if args.ls is not None:
        env_list = [e for e in viz.get_env_list() if args.ls in e]
        for env in env_list:
            print(env)

        exit(0)

    assert args.env_base_name is not None, "need env base name"
    assert args.feature_name is not None, "need feature name"
    envs = [e for e in viz.get_env_list() if args.env_base_name in e]
    assert len(envs) > 0, "no matching envs detected"

    # parse all envs
    json_blobs = [json.loads(viz.get_window_data(win=None, env=e)) for e in envs]
    print("found {} matching envs".format(len(json_blobs)))

    for env_name, env in zip(envs, json_blobs): # iterate over all different environments
        for window, value in env.items():       # over all windows in that env
            if 'title' in list(value.keys()):
                if value['title'] == args.feature_name:
                    assert len(value['content']['data']) == 1, "content has multiple items, how to handle?"
                    x = value['content']['data'][0]['x']
                    y = value['content']['data'][0]['y']
                    pd.DataFrame({'x': x, 'y': y}).to_csv(env_name + args.output, index=False)
