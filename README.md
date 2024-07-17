# A Noise-Oriented and Redundancy-Aware Instance Selection Framework

This repository contains a Python 3 implementation of the biO-IS Framework, currently under review at the ACM Transactions on Information Systems (TOIS) journal.

## Requirements

This project is based on ```python==3.6``` and it requires ```Docker version >= 24.0.1```.

## Installing

Clone this repository in your machine. Execute the installation under the settings directory.

```
git clone https://github.com/waashk/bio-is.git
cd bio-is/settings/
docker build -t bio:1.0 .
```

## Activating environment

```
cd ..
docker run --rm --name bio-is -p 8888:8888 -v `pwd`:/bio-is -i -t bio:1.0 /bin/bash
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Example 

```
python run_generateSplit.py -d <dataset> -m bio-is --datain <data_directory> --out <output_directory>;
```

or run the following jupyter notebook:

```
jupyter-notebook --port 8888 --ip=0.0.0.0 --allow-root
example.ipynb
```

or the following python example:

```
>>> from collections import Counter
>>> from sklearn.datasets import make_classification
>>> from src.main.python.iSel import bio
>>> X, y = make_classification(n_classes=2, class_sep=2,
... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
>>> print('Original dataset shape %s' % Counter(y))
Original dataset shape Counter({{1: 900, 0: 100}})
>>> selector = bio.BIOIS()
>>> idx = selector.sample_indices_
>>> X_train_selected, y_train_selected =  X_train[idx], y_train[idx]
>>> print('Resampled dataset shape %s' % Counter(y_train_selected))
Resampled dataset shape Counter({1: 36, 0: 14})
```

or run the following bash example:

```
bash bash/run_selection.sh
```

## Automatic Text Classification Datasets

| **Dataset**  | **Size** | **Dim.** | **# Classes** | **Density** | **Skewness**         | **Link**                                       |
|--------------|----------|----------|---------------|-------------|----------------------|------------------------------------------------|
| DBLP         | 38,128   | 28,131   | 10            | 141         | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555264) |
| Books        | 33,594   | 46,382   | 8             | 269         | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555256) |
| ACM          | 24,897   | 48,867   | 11            | 65          | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555249) |
| 20NG         | 18,846   | 97,401   | 20            | 96          | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555237) |
| OHSUMED      | 18,302   | 31,951   | 23            | 154         | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555276) |
| Reuters90    | 13,327   | 27,302   | 90            | 171         | Extremely Imbalanced | [LINK](https://doi.org/10.5281/zenodo.7555298) |
| WOS-11967    | 11,967   | 25,567   | 33            | 195         | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555385) |
| WebKB        | 8,199    | 23,047   | 7             | 209         | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555368) |
| Twitter      | 6,997    | 8,135    | 6             | 28          | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7554707) |
| TREC         | 5,952    | 3,032    | 6             | 10          | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555342) |
| WOS-5736     | 5,736    | 18,031   | 11            | 201         | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555379) |
| SST1         | 11,855   | 9,015    | 5             | 19          | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555319) |
| pang_movie   | 10,662   | 17,290   | 2             | 21          | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555283) |
| Movie Review | 10,662   | 9,070    | 2             | 21          | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555273) |
| vader_movie  | 10,568   | 16,827   | 2             | 19          | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555354) |
| MPQA         | 10,606   | 2,643    | 2             | 3           | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555268) |
| Subj         | 10,000   | 10,151   | 2             | 24          | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555339) |
| SST2         | 9,613    | 7,866    | 2             | 19          | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555310) |
| yelp_reviews | 5,000    | 23,631   | 2             | 132         | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555396) |
| vader_nyt    | 4,946    | 12,004   | 2             | 18          | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555361) |
| agnews       | 127,600  | 39,837   | 4             | 37          | Balanced             | [LINK](https://doi.org/10.5281/zenodo.7555424) |
| yelp_2013    | 335,018  | 62,964   | 6             | 152         | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555898) |
| imdb_reviews | 348,415  | 115,831  | 10            | 326         | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555547) |
| sogou        | 510,000  | 98,974   | 5             | 588         | Balanced             | [LINK](https://doi.org/10.5281/zenodo.5259056) |
| medline      | 860,424  | 125981   | 7             | 77          | Imbalanced           | [LINK](https://doi.org/10.5281/zenodo.7555820) |

To guarantee the reproducibility of the obtained results, all datasets and their respective CV train-test partitions are available on in the respective LINK column.

Each dataset contains the following files:
- texts.txt: Raw document set (text). One per line.
- score.txt: Document class whose index is associated with texts.txt
- splits/split_\<k\>.pkl:  pandas DataFrame with k-cross validation partition.
- tfidf/: the TFIDF representation for each fold in the CSR matrix format. (.gz)

## IS Input Representation 

The TF-IDF representation is used as input to all IS methods. Before creating the TFIDF matrix, we adopted the following steps as a pre-processing step: i. we removed stopwords using the standard list from the scikit-learn library [ 58 ] (version 0.23.2); and ii. we only kept features that appear in at least two documents.

## Output

The outputs of the run_generateSplit.py script are:

- saida_\<method\>.json: A JSON file containing general information about the execution, including i. reduction, ii. time, iii. hyperparameters, among others.
- split_\<nfolds\>_\<method\>.pkl: The new split train-test.
- split_\<nfolds\>_\<method\>_idxinfold.pkl: The training indexes considering the old partition. E.g. Through these indexes, you can directly access the respective documents in a specific fold of the TFIDF representation.


## Citation

```
@article{cunha2023sigir,
  title={An Effective, Efficient, and Scalable Confidence-Based Instance Selection Framework for Transformer-Based Text Classification},
  author={Cunha, Washington and Fran{\c{c}}a, Celso and Fonseca, Guilherme and Rocha, Leonardo and Gon{\c{c}}alves, Marcos Andr{\'e}},
  booktitle={Proceedings of the 46th international ACM SIGIR conference on Research and development in information retrieval},
  year={2023},
  doi = {https://dl.acm.org/doi/10.1145/3539618.3591638},
}

@article{cunha2023csur,
  title={A Comparative Survey of Instance Selection Methods applied to NonNeural and Transformer-Based Text Classification},
  author={Cunha, Washington and Viegas, Felipe and Fran{\c{c}}a, Celso and Rosa, Thierson and Rocha, Leonardo and Gon{\c{c}}alves, Marcos Andr{\'e}},
  journal={ACM Computing Surveys},
  year={2023},
  publisher={ACM New York, NY},
  issn = {0360-0300},
  doi = {https://dl.acm.org/doi/10.1145/3582000},
}
```

