"""Building blocks of the model and the shared statistics built on them.

The tensor/panel reshaping, the canonical coordinates that resolve the bilinear
scale and sign indeterminacy, the spectral-radius stability diagnostics, and the
constancy test with its chart scores, Nyblom-Hansen statistic, influence blocks,
Holm adjustment, simulated limit law and wild bootstrap.

Nothing in this package knows about the application panel or the experiment
design; both of those import from here.
"""

from __future__ import annotations
