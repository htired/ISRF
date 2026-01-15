# ISRF

## Examples to run the codes

1. Pretraining ({dataset}: beauty, sports, and toys.)

```bash
python pretrain.py ./data/{dataset}/ --cuda --batch_size 64 --checkpoint ./checkpoint/{dataset}/
```

2. Inference ({dataset}: beauty, sports, and toys.) 

```bash
python SR.py ./data/{dataset}/ --cuda --batch_size 16 --checkpoint ./checkpoint/{dataset}/
python DR.py ./data/{dataset}/ --cuda --batch_size 16 --checkpoint ./checkpoint/{dataset}/
```

