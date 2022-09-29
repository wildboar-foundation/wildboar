# Authors: Isak Samsten
# License: BSD 3 clause

import numpy as np

from wildboar.datasets.filter import make_str_filter


class TestStrFilter:
    def test_n_samples(self):
        f = make_str_filter("n_samples<100")
        assert f("hello", np.zeros((101, 10)), None) is False

    def test_dataset(self):
        f = make_str_filter("dataset=~TwoLead")
        assert f("TwoLeadECG", None, None) is True
