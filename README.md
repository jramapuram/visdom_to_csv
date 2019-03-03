# Visdom2CSV

A simple tool that pulls data from visdom and writes them to CSVs.

## Example Usage

Grab all environments with `CIFAR10_` in their name and with `test_accuracy` as their window names and saves to unique CSVs.

```bash
python main.py --visdom-url http://MYVISDOMINSTANCE \
               --visdom-port 8097 \
               --env-base-name CIFAR10_ \
               --feature-name test_accuracy
```

List all environments from the CLI with `CIFAR10_` in their name

```bash
python main.py --visdom-url http://MYVISDOMINSTANCE \
               --visdom-port 8097 \
               --ls CIFAR10_
```
