# Exact split-composition controls

Status: verified v0.1 control fixtures, not a classification of all zero
divisors or idempotents.

The named `algmul.Hs.v1` and `algmul.Os.v1` tables contain a declared
hyperbolic generator `g` with `g*g = 1`: `g=j` (basis index 2) for split
quaternions and `g=e4` (basis index 4) for split octonions. Therefore

`u = 1 + g`, `v = 1 - g`

are nonzero and isotropic, with `N(u)=N(v)=0` and `u*v=v*u=0`. These are exact
zero-divisor witnesses. They are not division-algebra results. The analogous
ordinary H and O controls have norm 2 and product 2, so this witness pattern is
not generalized to those fixtures.

The following block is executed by the test suite.

```python
from anyalgebra.fixtures.composition import (
    split_octonion_fixture,
    split_quaternion_fixture,
)
from anyalgebra.fixtures.composition_ops import composition_operations

controls = (
    (split_quaternion_fixture, 2),
    (split_octonion_fixture, 4),
)
report = {}
for factory, generator in controls:
    algebra = factory()
    operations = composition_operations(algebra)
    plus = algebra.module.element({0: 1, generator: 1})
    minus = algebra.module.element({0: 1, generator: -1})
    zero = algebra.module.zero()
    report[algebra.name] = {
        "generator": algebra.module.basis.labels[generator],
        "norms": [
            operations.quadratic_norm(plus).value,
            operations.quadratic_norm(minus).value,
        ],
        "products_are_zero": [
            operations.multiply(plus, minus) == zero,
            operations.multiply(minus, plus) == zero,
        ],
    }
```

Over QQ, the same witnesses give complementary idempotents

`p = (1 + g)/2`, `q = (1 - g)/2`.

The tests verify exactly that `p^2=p`, `q^2=q`, `p*q=q*p=0`, `p+q=1`, and
`N(p)=N(q)=0` for both split conventions. Division is explicitly refused at
zero norm. The package does not yet expose a general composition-algebra
inverse API, and this page makes no claim that the displayed witnesses exhaust
either split algebra.
