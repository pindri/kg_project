# Knowledge Graphs project

Project for the Knowledge Graphs (KG) course @TUWien.

## Project organisation

The code is organised as follows:
* `main.ipynb` presents the functionalities of the recommender system. It is presented in a notebook format, to provide a more convenient interaction.
* `data_builder.py` contains the code to create and load the KG;
* `factorisation.py` implements the Weighted Alternating Least Squares algorithm for matrix factorisation;
* `utilities.py` contains utility functions to measure the execution of functions.
* `data/` contains the data used for the recommendation task and to build the KG.
* `results/` contains an unorganised and incomplete collection of results from the experiments.
* `WALS/` contains the implementation of a matrix factorisation-based recommender system, adapted from a previous project.
* `KGCN/` contains an adaptation of [KGCN-pytorch](https://github.com/zzaebok/KGCN-pytorch) to work on this specific MovieLens knowledge graph.
* `report/` contains the portfolio for the project.
