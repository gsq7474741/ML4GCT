# Using Machine Learning Techniques for Evaluating Creativity in Graphic Combination Task (GCT)

This is the repository for the paper "Using Machine Learning Techniques for Evaluating Creativity in Graphic Combination Task (GCT)".

# R

Run `analysis.R` in RStudio to reproduce the results in the paper.

# Python
## Quick Start

Requires Python 3.10


### Conda
Install the required packages using Conda virtual environment:

#### Windows

```bash
conda create -n ag python=3.10
conda activate ag
conda install -c conda-forge mamba
mamba install -c conda-forge autogluon
mamba install -c conda-forge "ray-tune >=2.6.3,<2.7" "ray-default >=2.6.3,<2.7"  # install ray for faster training
pip install -r requirements.txt
```


#### Linux

```bash
conda create -n ag python=3.10
conda activate ag
conda install -c conda-forge mamba		
mamba install -c conda-forge autogluon
mamba install -c conda-forge "ray-tune >=2.6.3,<2.7" "ray-default >=2.6.3,<2.7"  # install ray for faster training
pip install -r requirements.txt
```

### Pip

Install the required packages using pip:

```bash
pip install -r requirements.txt
```


### Running the code

After installing the required packages, you can run the code by executing the following command:

```bash
python run_MLM.py
```

This will start the training process and print the mean metrics and t-test results table in our paper.

A sqlite database named `experiment.db` will be created in the directory. All the experiment details will be stored in this database.

If `dump_db` in the main function is set to `True`, the database will be dumped to csv files in the `dump` directory. 
These files can be opened using Excel or other tools for further analysis.

Existing experiment records (i.e., `experiment.db`, `dump` directory and `results.txt`) are the original results of the paper.