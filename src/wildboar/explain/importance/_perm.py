import math

import numpy as np
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.utils.validation import check_random_state

from wildboar.explain.importance.base import BaseImportance, Importance
from wildboar.utils import check_array


def _intervals(n, n_interval):

    for i in range(n_interval):
        length = n // n_interval
        start = i * length + min(i % n_interval, n % n_interval)
        if i % n_interval < n % n_interval:
            length += 1
        yield start, start + length


def _unpack_scores(orig_score, perm_score, intervals, n_timestep):
    importances = np.zeros((n_timestep, perm_score.shape[1]))
    for i, (start, end) in enumerate(intervals):
        importances[start:end, :] = orig_score - perm_score[i, :]

    return Importance(
        mean=np.mean(importances, axis=1),
        std=np.std(importances, axis=1),
        full=importances,
    )


class IntervalPermutationImportance(BaseImportance):
    def __init__(
        self,
        *,
        scoring=None,
        n_repeat=5,
        n_interval="sqrt",
        verbose=False,
        random_state=None,
    ):
        super().__init__(scoring=scoring)
        self.n_repeat = n_repeat
        self.n_interval = n_interval
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, estimator, x, y=None, sample_weight=None):
        x = check_array(x, allow_multivariate=False)
        y = check_array(y, ensure_2d=False)
        random_state = check_random_state(self.random_state)
        if x.shape[0] != y.shape[0]:
            raise ValueError()

        if self.n_interval == "sqrt":
            n_interval = math.ceil(math.sqrt(x.shape[-1]))
        elif self.n_interval == "log":
            n_interval = math.ceil(math.log2(x.shape[-1]))
        elif isinstance(self.n_interval, float):
            if not 0 < self.n_interval <= 1:
                raise ValueError("n_interval")
            n_interval = math.floor(x.shape[-1] * self.n_interval)
        elif isinstance(self.n_interval, int):
            n_interval = self.n_interval
        else:
            raise ValueError("unsupported n_interval, got %r" % self.n_interval)

        if callable(self.scoring):
            scoring = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scoring = check_scoring(estimator, self.scoring)
        else:
            scoring_dict = _check_multimetric_scoring(estimator, self.scoring)
            scoring = _MultimetricScorer(**scoring_dict)

        self.n_timestep_ = x.shape[-1]
        self.n_interval_ = n_interval
        self.intervals_ = list(_intervals(x.shape[-1], n_interval))
        scores = []
        for iter, (start, end) in enumerate(self.intervals_):
            if self.verbose:
                print(f"Running iteration {iter + 1} of {self.n_interval_}.")
            x_perm = x.copy()
            x_perm_interval = x_perm[:, start:end]
            rep_scores = []
            for rep in range(self.n_repeat):
                random_state.shuffle(x_perm_interval)
                x_perm[:, start:end] = x_perm_interval
                if sample_weight is not None:
                    score = scoring(estimator, x_perm, y, sample_weight=sample_weight)
                else:
                    score = scoring(estimator, x_perm, y)
                rep_scores.append(score)

            if isinstance(rep_scores[0], dict):
                scores.append(_aggregate_score_dicts(rep_scores))
            else:
                scores.append(rep_scores)

        if sample_weight is not None:
            self.baseline_score_ = scoring(estimator, x, y, sample_weight=sample_weight)
        else:
            self.baseline_score_ = scoring(estimator, x, y)

        if isinstance(self.baseline_score_, dict):
            self.importances_ = {
                name: _unpack_scores(
                    self.baseline_score_[name],
                    np.array([scores[i][name] for i in range(n_interval)]),
                    self.intervals_,
                    self.n_timestep_,
                )
                for name in self.baseline_score_
            }
        else:
            self.importances_ = _unpack_scores(
                self.baseline_score_,
                np.array(scores),
                self.intervals_,
                self.n_timestep_,
            )
        return self

    def score(self, timestep=None, dim=0):
        if timestep is not None:
            if isinstance(self.importances_, dict):
                return {
                    name: self.importances_[name].mean[timestep]
                    for name in self.importances_
                }
            else:
                return self.importances_.mean[timestep]
        else:
            return self.importances_

    def plot(self, ax=None, **kwargs):
        if "score" in kwargs:
            score = kwargs.pop("score")

        if isinstance(self.importances_, dict):
            ax.errorbar(
                np.arange(self.n_timestep_),
                self.importances_[score].mean,
                yerr=self.importances_[score].std,
            )
            ax.set_ylabel("Score (%.3f)" % self.baseline_score_[score])
        else:
            ax.errorbar(
                np.arange(self.n_timestep_),
                self.importances_.mean,
                yerr=self.importances_.std,
            )
            ax.set_ylabel("Score (%.3f)" % self.baseline_score_)
