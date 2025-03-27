# Authors: Isak Samsten
# License: BSD 3 clause
import math
import numbers

import numpy as np
from sklearn.utils import check_scalar

from ..distance._distance import _THRESHOLD, subsequence_match
from ..distance._matrix_profile import paired_matrix_profile
from ..utils.decorators import singleton
from ..utils.validation import check_array, check_option, check_type


@singleton
def motifs(  # noqa: PLR0912, PLR0915
    x,
    mp=None,
    *,
    window="auto",
    exclude=0.2,
    max_distance="auto",
    max_neighbors=10,
    min_neighbors=1,
    max_motifs=1,
    return_distance=False,
):
    """Find motifs.

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
    max_neighbors : int, optional
        The maximum number of neighbours.
    min_neighbors : int, optional
        The minimum number of neighbours.
    max_motifs : int, optional
        The maximum number of motifs to return.
    return_distance : bool, optional
        Return the distance from main to neighbours.

    Returns
    -------
    motif_indices : list
        List of arrays of motif neighbour indices
    motif_distance : list, optional
        List of arrays of distance from motif to neighbours

    See Also
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

        mp = paired_matrix_profile(
            x, window=window, exclude=exclude, return_index=False
        )
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

        mp = np.atleast_2d(mp).copy()
    else:
        raise TypeError(
            "matrix profile must be an ndarray, not %r" % type(mp).__qualname__
        )

    if max_neighbors is None:
        max_neighbors = x.shape[-1]

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
    motif_indices = []
    check_scalar(max_motifs, "max_motifs", numbers.Integral, min_val=1)
    check_scalar(min_neighbors, "min_neighbors", numbers.Integral, min_val=1)
    check_scalar(
        max_neighbors, "max_neighbors", numbers.Integral, min_val=min_neighbors
    )

    for i in range(x.shape[0]):
        motif_distance = []
        motif_index = []
        current_cutoff = max_distance
        if callable(max_distance):
            current_cutoff = max_distance(mp[i])

        while len(motif_index) < max_motifs and not np.all(np.isinf(mp[i])):
            candidate = np.argmin(mp[i])
            if mp[i, candidate] > current_cutoff:
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
                max_matches=max_neighbors,
                exclude=exclude,
                return_distance=True,
            )

            if match_idx.size > min_neighbors:
                motif_index.append(match_idx[:max_neighbors])
                motif_distance.append(match_dist[:max_neighbors])
                # The first match is always the same as candidate
                # so we can exclude all the matches from the matrix
                # profile
                for j in match_idx[:max_neighbors]:
                    start = max(0, j - exclude)
                    end = min(mp.shape[-1], j + exclude)
                    mp[i, start:end] = np.inf
            else:
                # Just exclude the candidate from the matrix profile
                start = max(0, candidate - exclude)
                end = min(mp.shape[-1], candidate + exclude)
                mp[i, start:end] = np.inf

        motif_distances.append(motif_distance)
        motif_indices.append(motif_index)

    if return_distance:
        return motif_indices, motif_distances
    else:
        return motif_indices
