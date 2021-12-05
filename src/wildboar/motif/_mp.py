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

from wildboar.distance import matrix_profile, subsequence_match
from wildboar.utils import check_array
from wildboar.utils.decorators import singleton


@singleton
def find_motifs(
    x,
    mp=None,
    window=None,
    exclude=None,
    max_distance=None,
    max_matches=10,
    min_neighbours=1,
    max_motif=1,
    cutoff=None,
    return_distance=True,
):
    if mp is None:
        if window is None:
            raise ValueError("if the matrix profile is not given, window must be set")

        mp, mpi = matrix_profile(x, window=window, exclude=exclude, return_index=True)
        mp = np.atleast_2d(mp)
        mpi = np.atleast_2d(mpi)
    elif isinstance(mp, (tuple, list)) and len(mp) == 2:
        mp, mpi = mp
        if not isinstance(mp, np.ndarray) and np.issubdtype(mp.dtype, np.double):
            raise ValueError("Unexpected matrix profile")

        if not isinstance(mpi, np.ndarray) and np.issubdtype(mp.dtype, np.intp):
            raise ValueError("Unexpected matrix profile index")

        if mp.shape != mpi.shape:
            raise ValueError("matrix profile and matrix profile index does not match")

        w = x.shape[-1] - mp.shape[-1] + 1
        if window is None:
            window = w
        elif window != w:
            raise ValueError("given window parameter is invalid, set to None")
    else:
        raise ValueError("unexpected matrix profile")

    if max_matches is None:
        max_matches = x.shape[-1]

    if cutoff is None:
        cutoff = max(np.mean(mp) - 2 * np.std(mp), np.min(mp))

    x = np.atleast_2d(x)
    x = check_array(x, dtype=np.double)

    motif_distances = []
    motif_indicies = []
    for i in range(x.shape[0]):
        candidates = np.argsort(mp[i])
        motif_distance = []
        motif_index = []
        for j in range(max_motif):
            candidate = candidates[j]
            if mp[i, candidate] > cutoff:
                break

            if isinstance(max_distance, float) and mp[i, candidate] > max_distance:
                break

            if len(motif_index) > max_motif:
                break

            match_idx, match_dist = subsequence_match(
                x[i, candidate : candidate + window].reshape(1, -1),
                x[i].reshape(1, -1),
                threshold=max_distance,
                metric="scaled_euclidean",
                max_matches=max_matches,
                return_distance=True,
            )

            if match_idx.size > min_neighbours:
                motif_index.append(match_idx[:max_matches])
                motif_distance.append(match_dist[:max_matches])

        motif_distances.append(motif_distance)
        motif_indicies.append(motif_index)

    if return_distance:
        return motif_indicies, motif_distances
    else:
        return motif_indicies
