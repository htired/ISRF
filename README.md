# Examples to run the codes

1. Pretraining ({dataset}: beauty, sports, and toys.)

```bash
python pretrain.py ./data/{dataset}/ --cuda --batch_size 64 --checkpoint ./checkpoint/{dataset}/
```

2. Inference ({dataset}: beauty, sports, and toys.) 

```bash
python seq_reranker.py ./data/{dataset}/ --cuda --batch_size 16 --checkpoint ./checkpoint/{dataset}/
python topn.py ./data/{dataset}/ --cuda --batch_size 16 --checkpoint ./checkpoint/{dataset}/
```

