# Educe

## Requirement

```
torch==1.10.0
numpy
pandas
```

The dataset should contain `fact.txt, train.txt, valid.txt, test.txt` in the format of `h r t` for each line, `entities.txt` listing all the entities shown in KG, `relations.txt` listing all the relations shown in KG, `attributes.txt` representing all the attributes it may be, and `attributes2values.txt` representing those can be connected by attributes in the format of `r entity`.

Run the following command to train Educe on your dataset:


## Train the model
Train the model with following command:
```shell
python src/main.py --datadir=datasets/DATASET/ --exp_name=DESC --use_value_vector=use_value_vector --batch_size=BATCH --num_step=NUM_STEP --rank=RANK --max_epoch=MAX_EPOCH --learning_rate=LR
```

## Decode rule

Decode rules from the model with following commands.
The parameters of threshold need to be adjusted with different datasets.

```shell
python src/main.py --datadir=datasets/DATASET/ --exp_name=DESC --use_value_vector=use_value_vector --batch_size=BATCH --num_step=NUM_STEP --rank=RANK --ckpt=CHECKPOINT_PATH --decode_rule
```

## Rule confidence

Open `confidence_Educe.ipynb` with `jupyter notebook` and change the dataset and rule path. Note that the rule nearly appears in the reasoning process will be removed.
