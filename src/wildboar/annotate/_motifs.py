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
import math
import numbers

import numpy as np

from wildboar.distance import _THRESHOLD, matrix_profile, subsequence_match
from wildboar.utils import check_array
from wildboar.utils.decorators import singleton


@singleton
def motifs(
    x,
    mp=None,
    window=None,
    exclude=0.2,
    max_distance="best",
    max_neighbours=10,
    min_neighbours=1,
    max_motif=1,
    return_distance=False,
):
    """Find motifs

    Parameters
    ----------
    x : array-like of shape (n_samples, n_timestep)
        The time series

    mp : ndarray or shape (n_samples, profile_size), optional
        The matrix profile. The matrix profile is computed if None.

    window : int, optional
        The window size of the matrix profile.

    exclude : float, optional
        The size of the exclusion zone.

    max_distance : str, optional
        The maximum distance between motifs.

    max_matches : int, optional
        The maximum number of neighbours

    min_neighbours : int, optional
        The minimum number of neighbours

    max_motif : int, optional
        The maximum number of motifs to return.

    return_distance : bool, optional
        Return the distance from main to neighbours

    Returns
    -------
    motif_indicies : list
        List of arrays of motif neighbour indicies

    motif_distance : list, optional
        List of arrays of distance from motif to neighbours

    See also
    --------
    wildboar.distance.subsequence_match : find subsequence matches

    wildboar.distance.matrix_profile : compute the matrix profile

    References
    ----------
    Yeh, C. C. M. et al. (2016).
        Matrix profile I: All pairs similarity joins for time series: a unifying view
        that includes motifs, discords and shapelets. In 2016 IEEE 16th international
        conference on data mining (ICDM)
    """
    if mp is None:
        if window is None:
            raise ValueError("if the matrix profile is not given, window must be set")

        mp = matrix_profile(x, window=window, exclude=exclude, return_index=False)
        mp = np.atleast_2d(mp)
    elif isinstance(mp, np.ndarray) and np.issubdtype(mp.dtype, np.double):
        w = x.shape[-1] - mp.shape[-1] + 1
        if window is None:
            window = w
        elif window != w:
            raise ValueError("given window parameter is invalid, set to None")

        mp = np.atleast_2d(mp).copy()
    else:
        raise ValueError("unexpected matrix profile")

    if max_neighbours is None:
        max_neighbours = x.shape[-1]

    if isinstance(max_distance, str):
        max_distance = _THRESHOLD.get(max_distance, None)
        if max_distance is None:
            raise ValueError("invalid max_distance (%r)" % max_distance)

    cutoff = max_distance
    if isinstance(exclude, numbers.Integral):
        if exclude < 0:
            raise ValueError("invalid exclusion (%d < 0)" % exclude)
    elif isinstance(exclude, numbers.Real):
        exclude = math.ceil(window * exclude)
    elif exclude is not None:
        raise ValueError("invalid exclusion (%r)" % exclude)

    x = check_array(np.atleast_2d(x), dtype=np.double)
    if x.shape[0] != mp.shape[0]:
        raise ValueError("not the same number of samples")

    motif_distances = []
    motif_indicies = []
    for i in range(x.shape[0]):
        motif_distance = []
        motif_index = []
        if callable(max_distance):
            cutoff = max_distance(mp[i])
        for j in range(max_motif):
            if len(motif_index) > max_motif:
                break

            candidate = np.argmin(mp[i])
            if mp[i, candidate] > cutoff:
                break

            if (
                isinstance(max_distance, numbers.Real)
                and mp[i, candidate] > max_distance
            ):
                break

            match_idx, match_dist = subsequence_match(
                x[i, candidate : candidate + window].reshape(1, -1),
                x[i].reshape(1, -1),
                threshold=max_distance,
                metric="scaled_euclidean",
                max_matches=max_neighbours,
                exclude=exclude,
                return_distance=True,
            )

            if match_idx.size > min_neighbours:
                motif_index.append(match_idx[:max_neighbours])
                motif_distance.append(match_dist[:max_neighbours])
                # The first match is always the same as candidate
                # so we can exclude all the matches from the matrix
                # profile
                for j in match_idx[:max_neighbours]:
                    start = max(0, j - exclude)
                    end = min(mp.shape[-1], j + exclude)
                    mp[i, start:end] = np.inf
            else:
                # Just exclude the candidate from the matrix profile
                start = max(0, candidate - exclude)
                end = min(mp.shape[-1], candidate + exclude)
                mp[i, start:end] = np.inf

        motif_distances.append(motif_distance)
        motif_indicies.append(motif_index)

    if return_distance:
        return motif_indicies, motif_distances
    else:
        return motif_indicies
