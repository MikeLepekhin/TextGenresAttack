# TextGenresAttack

In order to apply TextFooler for the Russian texts, it is necessary to load embeddings for Russian words. We use FastText embeddings for 30K most frequent words from the train dataset.

Code of finding the most frequent words and getting normalised embeddings for them is here:

* `prepare_en_fasttext_embeddings.ipynb`,

* `prepare_ru_fasttext_embeddings.ipynb`.

The process of training of the XLM-Roberta models with cross-validation for the English texts:

* `train_xlm_roberta_model_en_cross_val.ipynb` (for the English texts),

* `train_xlm_roberta_model_ru_cross_val.ipynb` (for the Russian texts).




All the attacking experiments were implemented with usage of the TextAttack framework.

[TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP](https://arxiv.org/abs/2005.05909).

```bibtex
@misc{morris2020textattack,
    title={TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP},
    author={John X. Morris and Eli Lifland and Jin Yong Yoo and Jake Grigsby and Di Jin and Yanjun Qi},
    year={2020},
    eprint={2005.05909},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

The number of the successfully attacked texts.

Hyperparameters of the TextFooler technique:

* $k$ — the number of the most similar words,

* $threshold$ -- the minimum dor product value between the USE emdeddings of the original and the attacked texts.


### Untargeted attacks

`threshold=0.84`

| Language\k | 15          | 30          | 50          |
|------------|-------------|-------------|-------------|
| EN         | 416 (32,9%) | 438 (34,7%) | 453 (35,8%) |
| RU         | 686 (47,4%) | 718 (49,6%) | 745 (51,5%) |


`threshold=0.6`

| Language\k | 15          | 30          | 50          |
|------------|-------------|-------------|-------------|
| EN         | 424 (33,5%) | 444 (35,1%) | 457 (36,2%) |
| RU         | 687 (47,5%) | 720 (49,8%) | 744 (51,4%) |

`threshold=0.4`

| Language\k | 15          | 30          | 50          |
|------------|-------------|-------------|-------------|
| EN         | 424 (33,5%) | 444 (35,1%) | 457 (36,2%) |
| RU         | 687 (47,5%) | 720 (49,8%) | 744 (51,4%) |

`threshold=0`

| Language\k | 15          | 30          | 50          |
|------------|-------------|-------------|-------------|
| EN         | 424 (33,5%) | 444 (35,1%) | 457 (36,2%) |
| RU         | 687 (47,5%) | 720 (49,8%) | 744 (51,4%) |


### Testing the robust models

`Language=RU`

| Model\k    | 15          | 30          | 50          |
|------------|-------------|-------------|-------------|
| base       | 363         | 373         | 375         |
| targeted   | 332         | 343         | 355         |
| robust     | 292         | 329         | 350         |


`Language=EN`

| Model\k    | 15          | 30          | 50          |
|------------|-------------|-------------|-------------|
| base       | 234         | 247         | 252         |
| targeted   | 254         | 269         | 272         |
| robust     | 209         | 234         | 244         |

### Targeted attacks

`threshold=0.84`

| Language\k | 15          | 30          | 50          |
|------------|-------------|-------------|-------------|
| EN         | 233 (34,2%) | 248 (36,4%) | 254 (37,2%) |
| RU         | 317 (57,3%) | 326 (59,0%) | 328 (59,3%) |

