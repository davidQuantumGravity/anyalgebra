# AnyAlgebra

AnyAlgebra is a Python framework for exact work with arbitrary algebraic
structures. You describe a structure directly (sorts, carriers, operations,
bases, structure constants) and the library evaluates expressions, checks
laws, finds counterexamples, and records replayable evidence, all in exact
integer or rational arithmetic.

It has no required dependencies and needs Python 3.11 or newer.

## Install

AnyAlgebra is not on a package index yet. Install it from a clone:

```powershell
git clone https://github.com/davidQuantumGravity/anyalgebra.git
cd anyalgebra
python -m venv .venv
.venv\Scripts\Activate.ps1   # on Linux or macOS: source .venv/bin/activate
python -m pip install -e .
```

## A first calculation

The octonions are available as a named exact multiplication table. This
computes one commutator and one associator in that table:

```python
from anyalgebra.analysis.elementary import associator, commutator
from anyalgebra.fixtures.composition import octonion_fixture

O = octonion_fixture()
e = [O.module.element({i: 1}) for i in range(O.module.rank)]

print(O.module.basis.labels)
print(commutator(O, e[1], e[2]).coordinates())
print(associator(O, e[1], e[2], e[4]).coordinates())
```

```text
('1', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7')
{3: _IntegerElement(parent=ZZ(), value=2)}
{7: _IntegerElement(parent=ZZ(), value=2)}
```

So `[e1, e2] = 2*e3`, and `(e1*e2)*e4 - e1*(e2*e4) = 2*e7`: the table is
neither commutative nor associative. Coordinates are exact integers keyed by
basis index.

The root package exports only `__version__`; import working APIs from their
owning modules, as above. Every file in [`examples/`](examples) is a runnable
script, for instance:

```powershell
python examples/octonion_counterexample.py
python examples/finite_algebra_atlas.py --output "$PWD/atlas-demo"
```

The atlas example writes into the folder named by `--output`. That path must
be absolute and must not exist yet.

## What can I do with it now?

The stable kernel can define arbitrary finite many-sorted structures, partial
operations, deductive systems, and exact finite multilinear algebras. It can
convert presentations, change bases, validate laws, find counterexamples,
recover structure constants from operators, compute bounded algebra
fingerprints, and save replayable evidence. Named generic fixtures cover
quaternions, split quaternions, octonions, and split octonions.

The v0.1 additions are exact sparse multilinear evaluation, bounded
finite-algebra censuses, constructive isomorphism evidence, capability-aware
analysis, and replayable content-addressed atlases, with an `anyalgebra-atlas`
command-line tool.

Where to go next:

- [What can AnyAlgebra do today?](docs/capabilities.md) maps common goals to
  runnable examples and confirmed outputs.
- The [finite algebra census and atlas manual](docs/guides/finite-algebra-census.md)
  walks through the v0.1 finite-operation workflow.
- The [quaternion and octonion guide](docs/guides/quaternions-and-octonions.md)
  covers composition algebras and catalogues every related test.
- The [v0.1 API](docs/api/api-v0.1.md) lists the normative module-level names.

### Experimental namespace

`anyalgebra.experimental` holds research code built on the kernel. The part
published here contains:

- an exact Albert-algebra and compact-F4 control in 27 coordinates;
- a complete division/split catalogue of the 2-by-2 and 3-by-3 magic
  squares, with exact standard orthogonal generators for every 2-by-2 entry;
- an exact derivation of the Baker--Campbell--Hausdorff series through degree
  four; and
- exact linear-algebra checks for tables of structure constants: the Jacobi
  identity on every basis triple, trace forms, inertia, and modular rank.

Constructions of exceptional Lie algebras built with these tools are being
prepared for publication and are not part of this repository yet.

Experimental modules are research code. They are not a stable API and may
change without notice.

### Not implemented yet

Stable constructors for Lie, Clifford, Jordan, `GL/SL/SO/SU(n, A)`, and
`A tensor J_n(B)` are not implemented. Version 0.1 is intentionally general:
it does not expose specialized public Lie, Clifford, Jordan, exceptional,
geometric, or physics APIs.

## Status

The source reports version `0.1.0`. All 60 planned v0.1 tasks are implemented
and the final task, `V01-060`, passed its readiness gate. Experimental work
done after that milestone does not promote the experimental API or change the
package version.

No Git tag, signed archive, or package-index publication exists yet. The
[current status](docs/status.md) page records the exact test, coverage, and
artifact evidence and its boundaries.

These are engineering results. They do not establish AlgMul correctness, a
general algebra classification, or a physics claim. No project/scientific claim
follows from a passing package or atlas gate.

## Development

```powershell
python -m pip install -e ".[dev]"
python -m ruff format --check src tests examples tools
python -m ruff check src tests examples tools
python -m mypy src tests
python -m pytest -q -m "not legacy and not optional_backend"
```

The neutral suite is large. For a quicker run, leave out the long exact
computations, which carry the `slow` marker:

```powershell
python -m pytest -q -m "not legacy and not optional_backend and not slow"
```

Tests in `tests/legacy` compare against a pinned legacy Mathematica package
and are excluded from the neutral run. A few tests audit maintainer process
records that are not part of this repository; they skip with an explicit
reason.

An equivalent UV-managed setup is:

```powershell
uv sync --extra dev
uv run python -m pytest -q -m "not legacy and not optional_backend"
```

Two release tools build temporary wheel and source distributions, clean-install
them, and reproduce the reference atlas from each. They do not retain or
publish anything. Both run `uv` and by default use only its local cache; add
`--allow-network` to let it download the build backend:

```powershell
python tools/check_v0_1_artifacts.py .
python tools/reproduce_v0_1_atlas.py .
```

The package must import with its required dependencies alone. Optional
adapters belong behind capability interfaces and stay out of the neutral-core
import path.

## Documentation

[`docs/README.md`](docs/README.md) is the index and defines which document
controls when two disagree. The main entries are the
[architecture decisions](docs/architecture/architecture-decisions/README.md),
the [conventions](docs/conventions/conventions-v0.0.md), the
[testing strategy](docs/testing/testing-strategy.md), the
[evidence model](docs/evidence/evidence-model.md), and the
[AlgMul parity contract](docs/legacy/algmul-parity.md).

## Repository layout

```text
pyproject.toml     # packaging and development-tool configuration
src/anyalgebra/    # the package; only names listed in the API docs are public
tests/             # observable contract tests
docs/              # contracts and guides
examples/          # runnable example scripts
tools/             # release, reproduction, and audit scripts
```

No module becomes public merely because it exists in `src/`. Some documents
refer to process records under `.agents/` and to research notes under
`docs/research/`; those are maintained privately and are not part of this
repository.

## License

AnyAlgebra is licensed under the
[GNU Affero General Public License, version 3 only](LICENSE)
(`AGPL-3.0-only`). The AGPL permits research, educational, nonprofit, and
commercial use subject to its terms. In particular, its source-sharing
conditions can apply when covered software is conveyed and when a modified
version is made available for users to interact with over a network. The
license text, rather than this summary, controls.

Organizations that want to incorporate AnyAlgebra into a proprietary
application or hosted service without using the resulting work under the
AGPL may ask the repository maintainer about a separate commercial license.
Academic and nonprofit requests may be considered individually. See
[Commercial licensing](COMMERCIAL-LICENSING.md) for the scope and contact
route. No proprietary-license permission is granted unless separate written
terms are agreed.

Unless a file or directory says otherwise, original material in this
repository is covered by the repository license. Third-party material remains
subject to any separate notices that accompany it.
