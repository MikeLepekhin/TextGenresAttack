Here we use the AllenNLP framework [AllenNLP: A Deep Semantic Natural Language Processing Platform](https://www.semanticscholar.org/paper/AllenNLP%3A-A-Deep-Semantic-Natural-Language-Platform-Gardner-Grus/a5502187140cdd98d76ae711973dbcdaf1fef46d).

```bibtex
@inproceedings{Gardner2017AllenNLP,
  title={AllenNLP: A Deep Semantic Natural Language Processing Platform},
  author={Matt Gardner and Joel Grus and Mark Neumann and Oyvind Tafjord
    and Pradeep Dasigi and Nelson F. Liu and Matthew Peters and
    Michael Schmitz and Luke S. Zettlemoyer},
  year={2017},
  Eprint = {arXiv:1803.07640},
}
```

We add the successfully attacked texts to the original train dataset and train XLM-Roberta on the data.
We train XLM-Roberta on each dataset 3 times with different random seeds and investigate how the hyperparameter $k$ affects the model accuracy on the test datasets. 

| Corpus\k                       | 0              | 15             | 30             | 50             |
|--------------------------------|----------------|----------------|----------------|----------------|
| LiveJournal                    | 0.76 +- 0.003  | 0.756 +- 0.008 | 0.755 +- 0.009 | 0.756 +- 0.005 |
| Genre-homogenous English texts | 0.747 +- 0.026 | 0.796 +- 0.011 | 0.771 +- 0.01  | 0.776 +- 0.029 |