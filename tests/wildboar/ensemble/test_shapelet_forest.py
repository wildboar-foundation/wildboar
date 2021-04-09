# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Authors: Isak Samsten

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from wildboar import datasets
from wildboar.ensemble import ShapeletForestClassifier, ShapeletForestRegressor


def test_shapelet_forest_classifier():
    x_test, x_train, y_test, y_train = load_dataset("GunPoint")
    clf = ShapeletForestClassifier(n_estimators=10, n_shapelets=10, random_state=1)
    clf.fit(x_train, y_train)
    # assert_almost_equal(1.0, clf.score(x_test, y_test))
    branches = [
        (
            [1, 2, -1, 4, -1, -1, 7, 8, -1, -1, -1],
            [6, 3, -1, 5, -1, -1, 10, 9, -1, -1, -1],
        ),
        ([1, -1, 3, 4, -1, -1, -1], [2, -1, 6, 5, -1, -1, -1]),
        ([1, -1, 3, -1, -1], [2, -1, 4, -1, -1]),
        ([1, 2, 3, -1, -1, 6, -1, -1, -1], [8, 5, 4, -1, -1, 7, -1, -1, -1]),
        ([1, 2, -1, 4, -1, -1, 7, -1, -1], [6, 3, -1, 5, -1, -1, 8, -1, -1]),
        (
            [1, -1, 3, -1, 5, 6, -1, -1, 9, -1, -1],
            [2, -1, 4, -1, 8, 7, -1, -1, 10, -1, -1],
        ),
        (
            [1, -1, 3, -1, 5, 6, -1, -1, 9, 10, -1, -1, -1],
            [2, -1, 4, -1, 8, 7, -1, -1, 12, 11, -1, -1, -1],
        ),
        (
            [1, -1, 3, -1, 5, -1, 7, -1, 9, -1, -1],
            [2, -1, 4, -1, 6, -1, 8, -1, 10, -1, -1],
        ),
        (
            [1, 2, -1, 4, -1, -1, 7, -1, 9, -1, -1],
            [6, 3, -1, 5, -1, -1, 8, -1, 10, -1, -1],
        ),
        ([1, 2, -1, 4, 5, -1, -1, -1, -1], [8, 3, -1, 7, 6, -1, -1, -1, -1]),
    ]
    thresholds = [
        (
            [
                0.2296377023891641,
                0.9155635629895074,
                0.29454396399363225,
                0.1373720604377242,
                0.7448065506203836,
            ],
            [
                0.2296377023891641,
                0.9155635629895074,
                0.29454396399363225,
                0.1373720604377242,
                0.7448065506203836,
            ],
        ),
        (
            [2.468504005311855, 1.9405258328324837, 0.8044381492031283],
            [2.468504005311855, 1.9405258328324837, 0.8044381492031283],
        ),
        (
            [5.196214986288551, 0.6456271511985696],
            [5.196214986288551, 0.6456271511985696],
        ),
        (
            [
                0.11979907699707767,
                7.3542586615363295,
                0.6272455273230206,
                0.6350912157721158,
            ],
            [
                0.11979907699707767,
                7.3542586615363295,
                0.6272455273230206,
                0.6350912157721158,
            ],
        ),
        (
            [
                2.784221806516613,
                1.1200408467290157,
                6.497136463351233,
                0.12808084029931077,
            ],
            [
                2.784221806516613,
                1.1200408467290157,
                6.497136463351233,
                0.12808084029931077,
            ],
        ),
        (
            [
                1.2903260832690235,
                0.43780123268050886,
                10.598683319564032,
                1.3793072079278341,
                0.11394946256420711,
            ],
            [
                1.2903260832690235,
                0.43780123268050886,
                10.598683319564032,
                1.3793072079278341,
                0.11394946256420711,
            ],
        ),
        (
            [
                0.8027463521571572,
                0.09565984718772223,
                2.89080272806431,
                2.7213845181742036,
                2.5475295341730972,
                0.06999681987107359,
            ],
            [
                0.8027463521571572,
                0.09565984718772223,
                2.89080272806431,
                2.7213845181742036,
                2.5475295341730972,
                0.06999681987107359,
            ],
        ),
        (
            [
                0.6168414923246643,
                1.0177285031418821,
                0.0,
                0.6296623493515471,
                0.09161149477853166,
            ],
            [
                0.6168414923246643,
                1.0177285031418821,
                0.0,
                0.6296623493515471,
                0.09161149477853166,
            ],
        ),
        (
            [
                2.9552062212994716,
                0.6052276376662066,
                3.703067227838532,
                0.6760648548690982,
                0.47734283050609966,
            ],
            [
                2.9552062212994716,
                0.6052276376662066,
                3.703067227838532,
                0.6760648548690982,
                0.47734283050609966,
            ],
        ),
        (
            [
                7.142302010755662,
                1.0892908550023375,
                0.9449047717483255,
                0.017213098330187426,
            ],
            [
                7.142302010755662,
                1.0892908550023375,
                0.9449047717483255,
                0.017213098330187426,
            ],
        ),
    ]
    for estimator, (left, right), (left_threshold, right_threshold) in zip(
        clf.estimators_, branches, thresholds
    ):
        assert_equal(left, estimator.tree_.left)
        assert_equal(right, estimator.tree_.right)
        assert_almost_equal(
            left_threshold, estimator.tree_.threshold[estimator.tree_.left > 0]
        )
        assert_almost_equal(
            right_threshold, estimator.tree_.threshold[estimator.tree_.right > 0]
        )


def load_dataset(name):
    x_train, x_test, y_train, y_test = datasets.load_dataset(
        name,
        repository="wildboar/ucr-tiny",
        cache_dir="wildboar_datasets_cache",
        create_cache_dir=True,
        merge_train_test=False,
    )
    return x_test, x_train, y_test, y_train
