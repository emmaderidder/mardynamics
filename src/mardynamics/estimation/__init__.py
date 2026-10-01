"""The MAR(1) estimators.

``mar_fit``
    The tensor-facing estimators: the static NumPy projection-initialised
    alternating-least-squares fit and the regime-switching estimator with its
    one-step forecast rule.
``regime_em``
    The EM machinery the regime-switching estimator runs: the weighted ALS
    update shared by every active MAR-LSE route, the static warm start with its
    residual-based regime initialisation, and the Gaussian hidden-Markov
    filter/smoother.
"""

from __future__ import annotations
