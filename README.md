# Triplet Contrastive Learning with Learnable Sequence Augmentation for Sequential Recommendation

## Introduction
The quality of augmented data directly affects the performance of contrastive learning. 
Low-quality augmentation offers limited benefits for model optimization. 
Existing contrastive learning-based sequential recommendation works primarily utilize heuristic data augmentation methods, 
which often exhibit excessive randomness and struggle to generate positive samples that align with users' true intentions.

To address this limitation, we propose Triplet Contrastive Learning with Learnable Sequence Augmentation for Sequential Recommendation. 
Unlike heuristic augmentation methods, we train a learnable sequence augmentation module that can automatically leverage the self-supervised information from global context to select appropriate modification positions and augmentation operations, generating positive samples that more accurately reflect user preferences.
Furthermore, we design a ranking-based triplet contrastive loss to enhance the positive feedback from augmented sequence generated by the augmentation module, providing more nuanced contrastive signals for model optimization.
We conduct extensive experiments on three real-world datasets and the experimental results demonstrate that TCLARec outperforms state-of-the-art sequential recommendation baselines. 
Our in-depth analyses confirm that both the learnable augmentation and triplet contrastive learning contribute to improving recommendation accuracy.


## Datasets

We use [Beauty](http://jmcauley.ucsd.edu/data/amazon/links.html), [Sports and outdoors](http://jmcauley.ucsd.edu/data/amazon/links.html) and [Yelp](https://www.yelp.com/dataset) datasets for experiments. 
if you download raw datasets from official websites, please refer to ./dataprocessing/readme.md for the details about dataset processing.

## Experiments

For training, validating and testing model, please run:

```python
python main_traincr.py -m=train
python main_traincr.py -m=valid
python main_traincr.py -m=test
```

For other datasets, please revise the path of dataset and item_num in *main_traincr.py*.

If you want to set the probabilities of keep, delete and insert for generating randomly modified sequences when training the augmentation module, please revise the plist in *main_traincr.py*.

If you want to set the epochs for pre-training augmentation module, please revise the pre-training epoch in *main_traincr.py*.
Note that the pre-training epoch must be less than the epoch.

Tips:
If there is no such large resource, you need to set a smaller batch size in *main_traincr.py*. 
