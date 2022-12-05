# Authors: Isak Samsten
# License: BSD 3 clause
import math
import numbers

import numpy as np
from sklearn.utils import check_scalar

from ..distance import _THRESHOLD, matrix_profile, subsequence_match
from ..utils.decorators import singleton
from ..utils.validation import check_array, check_option, check_type


@singleton
def motifs(
    x,
    mp=None,
    *,
    window="auto",
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

    window : "auto", int or float, optional
        The window size of the matrix profile.

        - if "auto" the window is math.ceil(0.1 * n_timesteps) if mp=None, and the
          window of the matrix profile if mp is not None.
        - if float, a fraction of n_timestep
        - if int, the exact window size

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
        if window == "auto":
            window = 0.1

        mp = matrix_profile(x, window=window, exclude=exclude, return_index=False)
        mp = np.atleast_2d(mp)
    elif isinstance(mp, np.ndarray) and np.issubdtype(mp.dtype, np.double):
        w = x.shape[-1] - mp.shape[-1] + 1
        if window == "auto":
            window = w
        elif window != w:
            raise ValueError(
                "window == %r, expecting window == %d. Set window='auto', to "
                "automatically set window from the supplied matrix profile."
                % (window, w)
            )
        else:
            raise ValueError("window must be 'auto' or float, got %r" % window)

        mp = np.atleast_2d(mp).copy()
    else:
        raise TypeError(
            "matrix profile must be an ndarray, not %r" % type(mp).__qualname__
        )

    if max_neighbours is None:
        max_neighbours = x.shape[-1]

    if isinstance(max_distance, str):
        max_distance = check_option(_THRESHOLD, max_distance, "max_distance")

    cutoff = max_distance

    check_type(exclude, "exclude", (numbers.Integral, numbers.Real), required=False)
    if isinstance(exclude, numbers.Integral):
        exclude = check_scalar(
            exclude, "exclude", numbers.Integral, min_val=0, max_val=window
        )
    elif isinstance(exclude, numbers.Real):
        exclude = math.ceil(
            window
            * check_scalar(
                exclude,
                "exclude",
                numbers.Real,
                min_val=0,
                max_val=1,
                include_boundaries="right",
            )
        )

    x = check_array(np.atleast_2d(x), dtype=np.double)
    if x.shape[0] != mp.shape[0]:
        raise ValueError(
            "The matrix profile and x does not have the same number of samples. "
            "Set mp=None, to correctly compute the matrix profile for x."
        )

    motif_distances = []
    motif_indicies = []
    for i in range(x.shape[0]):
        motif_distance = []
        motif_index = []
        if callable(max_distance):
            cutoff = max_distance(mp[i])

        while len(motif_index) < max_motif and not np.all(np.isinf(mp)):
            candidate = np.argmin(mp[i])
            if mp[i, candidate] > cutoff:
                break

            if (
                isinstance(max_distance, numbers.Real)
                and mp[i, candidate] > max_distance
            ):
                break

            match_idx, match_dist = subsequence_match(
                x[i, candidate : candidate + window],
                x[i],
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
