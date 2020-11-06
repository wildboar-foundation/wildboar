# wildboar

[wildboar](https://isaksamsten.github.io/wildboar/) is a Python module for temporal machine learning and fast
distance computations built on top of
[SciKit-Learn](https://scikit-learn.org) and [Numpy](https://numpy.org)
distributed under the GNU General Public License Version 3.

It is currently maintained by Isak Samsten

## Installation

### Dependencies

wildboar requires:

 * Python (>= 3.6)
 * NumPy (>= 1.8.2)
 * SciPy (>= 0.13.3)
 
Some parts of wildboar is implemented using Cython. Hence, compilation
requires:

 * Cython (>= 0.28)

### Current version

- Current release: 1.0
- Current development release: 1.0.0dev

### Binaries

`wildboar` is available through `pip` and can be installed with:

    pip install wildboar

Universal binaries are compiled for GNU/Linux and Python 3.6 and 3.7.

### Compilation

If you already have a working installation of NumPy, SciPy and Cython,
compiling and installing wildboar is as simple as:

    python setup.py install
	
To install the requirements, use:

    pip install -r requirements.txt
	

## Development

Contributions are welcome. Pull requests are encouraged to be
formatted according to
[PEP8](https://www.python.org/dev/peps/pep-0008/), e.g., using
[yapf](https://github.com/google/yapf).

## Usage

    from wildboar import ShapeletForestClassifier
    from wildboar.dataset import load_two_lead_ecg
    x, y = load_two_lead_ecg()
    c = ShapeletForestClassifier()
    c.fit(x, y)

## Source code

You can check the latest sources with the command:

    git clone https://github.com/isakkarlsson/wildboar
	
## Citation

If you use wildboar in a scientific publication, I would appreciate
citations to the paper: Karlsson, I., Papapetrou, P. Boström, H.,
*Generalized Random Shapelet Forests*. In the Data Mining and
Knowledge Discovery Journal (DAMI), 2016
