## Setup

#### Using conda
```bash
conda env create --prefix ./venv --file env.yaml
```

This will rely on the `nphil` dependency list to install `numpy`, `scikit`,
and the likes.

#### By cloning this directory

```bash
# Note: python3 required. For example:
# conda create --prefix "./venv" python=3.10
# conda activate ./venv
conda install -c conda-forge mkl mkl-include
git clone https://github.com/capoe/nphil.git
cd nphil
pip install .
```

## Example

```bash
# (Make sure pip/conda environment is activated)
cd example
python generate_toy_dataset.py
philter --extt_file example.extt
```

## ExtendedTxt (.extt) format

```
ln 1: Array specification in the form array_name:(n_rows,n_cols). Example:
  X:(100,10) Y:(100,)
ln 2: Json string with metadata. Example:
  {"description": "Wow, this dataset is *really* noisy"}
ln 3 onwards: Arrays in numpy txt format, in order corresponding to header. Example:
  0.00 1.00 2.00 3.00 4.00 5.00 6.00 7.00 8.00 9.00
  0.01 1.01 2.01 3.01 4.01 5.01 6.01 7.01 8.01 9.01
  ...
  ...
  0.99 1.99 2.99 3.99 4.99 5.99 6.99 7.99 8.99 9.99
  0.00
  0.01
  ...
  ...
  0.99
```

