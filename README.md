# implementations

From-scratch machine-learning implementations by [Andreas Hermann](https://github.com/andreas-he) — the
reviewer-facing home for the implementation block of a 9-month AI-safety upskill plan.

The rule for everything here: **no library does the part being learned.** Gradients are derived and
written out, optimizers are hand-rolled, and every notebook ends with `assert`s that check the result
against a reference implementation (scikit-learn or PyTorch) rather than against a plot that looks right.

## Scope

**Phase 1 — classical ML from scratch (Aug–Oct 2026)**

| Topic | Built with | Checked against |
|---|---|---|
| Linear regression (closed form + gradient descent) | numpy | `sklearn.linear_model.LinearRegression` |
| Logistic regression (binary, then softmax) | numpy | `sklearn.linear_model.LogisticRegression` |
| k-nearest neighbours | numpy | `sklearn.neighbors.KNeighborsClassifier` |
| Backprop through an MLP (no autograd) | numpy | PyTorch autograd on the same weights |

**Phase 2 — deep-learning internals (Oct 2026 onward)**: optimizers, a transformer forward/backward
pass, PPO, and a sparse autoencoder, each in the same from-scratch-then-assert shape.

## Layout

```
<topic>/
  notebook.ipynb   # derivation, implementation, asserts
  README.md        # what was hard, what the asserts check
```

One folder per topic. Each folder stands alone; there is no shared package to install beyond
`numpy`, `scikit-learn`, `torch` and `jupyter`.

## Status

Created 2026-09-16 and currently empty of notebooks. The working copies live in a private practice
repo (`ai-safety-challenges/implementations/`) while each topic is in progress; a topic is copied
here once its asserts pass and its README is written, so that everything in this repo is finished
work rather than work in flight.
