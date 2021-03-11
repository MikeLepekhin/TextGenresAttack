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

