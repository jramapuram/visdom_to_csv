# Visdom2CSV

A simple tool that pulls data from visdom and writes them to CSVs.

## Example Usage

Grab **ALL** environments with `CIFAR10_` in their name and with `test_accuracy` as their window names and saves to unique CSVs.

```bash
python vis2csv.py --visdom-url http://MYVISDOMINSTANCE \
                  --visdom-port 8097 \
                  --env-base-name CIFAR10_ \
                  --feature-name test_accuracy
```

List all environments from the CLI with `CIFAR10_` in their name

```bash
python vis2csv.py --visdom-url http://MYVISDOMINSTANCE \
                  --visdom-port 8097 \
                  --ls CIFAR10_
```


## Example vis2png Usage

Writes a PNG with variance plots using `seaborn`.

``` bash
python vis2png.py --visdom-url http://MYVISDOMINSTANCE \
                  --visdom-port 8097 \
                  --env-base-name ENV1 ENV2 \
                  --feature-names FEATURE1 FEATURE2 ... \
                  --y-label="negative elbo" \
                  --title="MYTITLE" \
                  --output MYIMAGE.png \
                  --pickle-output MYPLT.pkl \
                  --legends MYCUSTOMLEGEND1 MYCUSTOMLEGEND2 \
                  --legend-features MYCUSTOMLEGENDPOSTFIX1 MYCUSTOMLEGENDPOSTFIX2 \
                  --y-range 105 130
```
