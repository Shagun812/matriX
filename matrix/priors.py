"""B3b prior: multinomial logistic regression on the SAME encoded features the LLM sees.

Train on >= 5,000 simulated cases from the dev-likelihood stream (NOT the 40 dev cases).
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import encode
from .schemas import HYPOTHESES, AgentVisible


class PriorModel:
    training_domain = "synthetic"
    synthetic_data_used = True

    def __init__(self, C: float = 1.0):
        self.pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        self.fitted = False

    def fit(self, visibles: Sequence[AgentVisible], labels: Sequence[str]) -> "PriorModel":
        if len(visibles) < 5000:
            warnings.warn(f"B3b trained on {len(visibles)} < 5000 cases; the README control requires >= 5000.")
        X = np.stack([encode(v) for v in visibles])
        y = np.array([HYPOTHESES.index(l) for l in labels])
        self.pipe.fit(X, y)
        self.fitted = True
        return self

    def predict(self, visible: AgentVisible) -> np.ndarray:
        proba = self.pipe.predict_proba(encode(visible)[None, :])[0]
        out = np.zeros(len(HYPOTHESES))
        out[self.pipe[-1].classes_] = proba  # classes missing from training stay 0
        return out / out.sum()

    def save(self, path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path) -> "PriorModel":
        return pickle.loads(Path(path).read_bytes())


def main(argv=None) -> int:
    """python -m materialfalsify.priors train --n 5000 --seed 4 --out artifacts/prior.pkl [--mock]"""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--out", default="artifacts/prior.pkl")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args(argv)
    if a.mock:
        from . import mock
        vis, labels = mock.training_data(a.n, a.seed)
    else:
        from .sim import generate_visible  # Person B: (n, seed, stream="dev_likelihood") -> (visibles, labels)
        vis, labels = generate_visible(a.n, a.seed, stream="dev_likelihood")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    PriorModel().fit(vis, labels).save(a.out)
    print(f"trained B3b prior on {len(vis)} cases -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())