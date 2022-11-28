
# PC-Attack

Source code for AAAI'23 paper "Practical Cross-system Shilling Attacks with Limited Access to Data".

## Environment
- Linux with Python ≥ 3.6
- PyTorch ≥ 1.4.0
- 0.5 > DGL ≥ 0.4.3

Use the command  "conda env create  -n env_pc_attack -f environment.yml" to copy the exact same environment.

## Data

The datasets used in our experiments can be found in the [data](../data) folder.

We use datasets that are widely used in previous [work](https://github.com/XMUDM/ShillingAttack).


## Command Line Parameters
`run.py` is the main entry of the program, it requires several parameters:

- `dataset`: the source dataset used in the experiment (Possible values:  ''filmtrust'', ''automotive'', "yelp", ''ToolHome''.  Default is  "yelp").
- `target-dataset`: the target dataset used in the experiment (Possible values:  ''filmtrust'', ''automotive'', "yelp", ''ToolHome''.  Default is  "filmtrust").
- `target-item`: id of the target item (Default is 5).
- `epochs`: training rounds.
- `gpu`:  GPU id.

## Examples

Please refer to `run.sh` for some running examples.

```
python run.py --dataset yelp --target-dataset filmtrust --target-item 5 --epochs 64 --gpu 2
```

