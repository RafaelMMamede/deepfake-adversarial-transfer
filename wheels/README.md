# Bundled AutoAttack wheel

This directory contains the AutoAttack wheel used by the released experiment
environment:

`autoattack-0.1-py3-none-any.whl`

- **Upstream project:** https://github.com/fra31/auto-attack
- **Upstream authors:** Francesco Croce and Matthias Hein
- **Upstream license:** MIT
- **SHA-256:** `f5a3e2641abfca1af0b4548d11f2e3e0a00d11df9c07ae8c9de054dccb56e6ac`

## Compatibility patch

The bundled wheel preserves the upstream AutoAttack implementation except for
one Python 3.7 compatibility change in `autoattack/autopgd_base.py`.

Upstream uses:

```python
n_fts = math.prod(self.orig_dim)
```

`math.prod` was introduced in Python 3.8, while the experiment environment
uses Python 3.7. The bundled wheel replaces that expression with the equivalent:

```python
from functools import reduce
from operator import mul

n_fts = reduce(mul, self.orig_dim, 1)
```

No attack hyperparameters or algorithmic behavior are changed by this patch.

The wheel itself includes the upstream MIT `LICENSE` file. The repository-level
`THIRD_PARTY_NOTICES.md` also records this attribution.
