# This file is part of wildboar
#
# wildboar is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wildboar is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Isak Samsten

import numpy as np
from wildboar.datasets._filter import make_str_filter


class TestStrFilter:
    def test_n_samples(self):
        f = make_str_filter("n_samples<100")
        assert f("hello", np.zeros((101, 10)), None) == False

    def test_dataset(self):
        f = make_str_filter("dataset=~TwoLead")
        assert f("TwoLeadECG", None, None) == True
