## Setup

```bash
conda create --prefix "./venv" python=3.7
conda activate "./venv"
conda install -c intel mkl mkl-include libboost
pip install .
```

## Example

```bash
cd example
