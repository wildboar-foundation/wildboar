# Authors: Isak Samsten
# License: BSD 3 clause
import numpy as np
from wildboar.datasets import load_two_lead_ecg
from wildboar.transform import PAA, SAX


def test_sax():
    X, _ = load_two_lead_ecg()
    X_sax = SAX(n_intervals=10, n_bins=5).fit_transform(X)

    assert X_sax.dtype == np.uint8
    np.testing.assert_equal(
        X_sax[0], np.array([3, 3, 2, 1, 0, 1, 3, 3, 3, 3], dtype=np.uint8)
    )


def test_paa():
    X, _ = load_two_lead_ecg()
    X_paa = PAA(n_intervals=10).fit_transform(X)

    assert X_paa.dtype == X.dtype
    np.testing.assert_equal(
        X_paa[0],
        np.array(
            [
                0.6462012761169009,
                0.3195832719405492,
                0.15792051772587,
                -0.6629047906026244,
                -2.3549310117959976,
                -0.39324739947915077,
                0.44239426776766777,
                0.6320434473454952,
                0.7861334010958672,
                0.30608393251895905,
            ]
        ),
    )
