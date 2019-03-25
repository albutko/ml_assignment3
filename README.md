github repository with code can be found: https://github.com/albutko/ml_assignment3

This project was completed using Python 3.7 and Jupyter Notebooks

To install necessary dependencies and `environment.yml` file has been provided that creates a conda environment named `alex_butenko_env`

Make sure to install Anaconda before going further.

Create conda virtualenv:
`conda env create -f environment.yml`

Then activate the environment:
`conda activate alex_butenko_env`

The project is broken into 4 jupyter notebooks stored in the `code` directory:
* Clustering.ipynb - Part 1 of HW3
* Dimensionality Reduction.ipynb - Part 2 of HW 3
* Clustering+Dimensionality Reduction.ipynb - Part 3 of HW3
* NNet Training.ipynb - Part 4/5 of HW3

To reproduce any results from the paper, open the appropriate jupyter notebook and run the entire notebook. If you wish to run one cell specifically, make sure to run the correct data cell to set the data variable to the correct datasets

Each juptyer notebook is labeled with headings to denote what dataset, clustering algorithm, of Dimensionality reduction algorithm is to be analyzed in the section

To open a single juptyer notebook run the command:
`jupyter notebook /path/to/notebook.ipynb`

Then to run a cell hit Ctrl-Enter
