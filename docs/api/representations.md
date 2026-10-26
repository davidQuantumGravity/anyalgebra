# Representations, branching and subalgebras

Status: implemented in the v0.1.10 patch for complex Lie algebras. Real
forms are covered only by a table of maximal compact subalgebras; see the
limits at the end.

```python
import anyalgebra.branching as ab
import anyalgebra.weights as aw
```

## Types and weights

A type is a sum of simple factors and `U1` factors: `aw.lie_type("E6")`,
`aw.lie_type("A2+A1+U1")`. A weight is a tuple: the Dynkin labels of each
simple factor in the Bourbaki numbering, in order, then one charge per `U1`.
An irreducible representation is named by its highest weight.

```python
E6 = aw.lie_type("E6")
first = (1, 0, 0, 0, 0, 0)
print(E6.dimension(first))                       # 27
print(E6.show(E6.tensor(first, first)))          # 351 + 351' + 27*
print(E6.show(E6.symmetric_power(first, 3)))     # 3003 + 650 + 1
print(E6.reality(first), E6.index(first))        # complex 6
```

| Call | Result |
|---|---|
| `T.dimension(w)` | Weyl's dimension formula |
| `T.character(w)`, `T.dominant_character(w)` | every weight, or the dominant ones, with multiplicities (Freudenthal) |
| `T.tensor(v, w)` | the content of a tensor product (Klimyk) |
| `T.symmetric_power(w, k)`, `T.exterior_power(w, k)` | the content of a power, for any `k` |
| `T.decompose(character)` | the irreducible content of a set of weights |
| `T.dual(w)`, `T.reality(w)` | the dual; `real`, `pseudoreal` or `complex` |
| `T.casimir(w)`, `T.index(w)` | `(w, w + 2 rho)` and the Dynkin index, 1 for the defining representation of `A_n` |
| `T.orbit(w)`, `T.to_dominant(w)`, `T.inner(v, w)` | the Weyl group and the invariant form, long roots of squared length 2 |
| `T.label(w)`, `T.show(content)` | names by dimension, such as `(3*,2)(1)` |

A representation is named by its dimension. A complex representation whose
highest weight is smaller than that of its dual, in the order of tuples, is
marked `*`; a second representation with the same name gets a prime. These
are conventions of this package: the tables in the literature differ among
themselves on which member of a pair carries the bar.

Any representation can be computed. The cost grows with the number of
weights, and the tests cover small ones.

## Global questions

The centre of the simply connected group is the weight lattice modulo the
root lattice. It is computed from the Cartan matrix by Smith normal form.

```python
D4 = aw.lie_type("D4")
vector, spinor, adjoint = (1, 0, 0, 0), (0, 0, 0, 1), (0, 1, 0, 0)
print(D4.center())                               # (2, 2)
print(D4.congruency(vector), D4.congruency(spinor))
print(D4.fundamental_group([vector]))            # 2: the group SO(8)
print(D4.fundamental_group([vector, spinor]))    # 1: the group Spin(8)
print(D4.fundamental_group([adjoint]))           # 4: the adjoint group
```

| Call | Answers |
|---|---|
| `T.center()` | the orders of the cyclic factors of the centre |
| `T.congruency(w)` | the class of a weight; zero exactly on the root lattice, and additive |
| `T.acting_center(reps)` | how much of the centre acts nontrivially on some representations |
| `T.fundamental_group(reps)` | the order of the fundamental group of the group that acts faithfully on them |

So a set of representations determines the global form of the group: the
one on which they are all single-valued and together faithful.

## Subalgebras and branching

An embedding of `h` in `g` is a projection matrix from weights of `g` to
weights of `h`. Branching projects the weights of a representation and
decomposes the result.

```python
for step in ab.maximal_subalgebras(E6):
    print(step.target.name, "|", step.show(first))
```

No projection matrix is typed in. Each is derived in one of four ways.

| Call | Subalgebra |
|---|---|
| `ab.regular(T, node)` | delete a node of the extended Dynkin diagram: semisimple of the same rank |
| `ab.regular(T, node, extended=False)` | delete a node of the diagram and keep a `U1`: a Levi subalgebra with its charge |
| `ab.via_representation(sub, [w, ...])` | `sub` inside `sl(N)`, `so(N)` or `sp(N)` by an `N`-dimensional representation |
| `ab.principal_sl2(T)`, `ab.folding(T)` | the principal `A1`; `F4` in `E6` and `G2` in `D4` |
| `ab.from_subalgebra(g, vectors)` | any subalgebra given by explicit elements of a Lie algebra |
| `ab.on_factor(T, k, step)` | an embedding applied to one simple factor of a larger type |

Nodes count from zero in the Bourbaki numbering; the extra node of the
extended diagram has the number `rank`. `ab.marks(T)` gives the coefficients
of the highest root. Deleting a node with a prime mark from the extended
diagram gives a maximal subalgebra, and so does replacing a node with mark 1
by a `U1` (Borel and de Siebenthal).

`ab.from_subalgebra` is the one that does not rest on any table. It extends
a Cartan subalgebra of the subalgebra to one of the whole algebra and
expresses the coroots of one in the coroots of the other, so a branching
rule follows from structure constants alone:

```python
import anyalgebra.lie as al

so5 = al.so(5)
inside = [
    tuple(int(i == k) for i in range(so5.dimension))
    for k, label in enumerate(so5.labels)
    if "5" not in label
]
step = ab.from_subalgebra(so5, inside)
print(step.source.name, ">", step.target.name, "|", step.show((0, 1)))
```

An embedding has `step.branch(w)`, `step.show(w)`, `step.project(weight)`,
`step.then(next_step)` and `step.index()`, the embedding index of each
simple factor.

## Chains

```python
D5 = aw.lie_type("D5")
for chain in ab.chains(D5, aw.lie_type("A2+A1+U1+U1"), max_steps=2):
    print(chain.label, "|", chain.show((0, 0, 0, 0, 1)))
```

`ab.chains(source, target, max_steps)` walks down through
`ab.maximal_subalgebras` and returns every composite embedding that reaches
the target type; `chain.label` records the path. `ab.regular_subalgebras(T)`
lists the types of all regular semisimple subalgebras by Dynkin's procedure.

## Real forms

`ls.maximal_compact(name)` gives the type of the maximal compact subalgebra
of a named real form, for example `F4` for `e6(-26)` and `B4` for `so(9,1)`.
Together with `ls.identify`, an algebra given by structure constants can be
named and its maximal compact subalgebra read off. The tests check, for
every real form of fourteen types, that the dimension of this subalgebra is
the one the character requires.

## Limits

- `ab.maximal_subalgebras` lists the regular maximal subalgebras, the
  principal `A1` where it is maximal, `F4` in `E6`, and the classical cases
  `so(n)` and `sp(n)` in `sl(n)`, `so(2n-1)` in `so(2n)` and `G2` in
  `so(7)`. It does not list the tensor-product subalgebras of the classical
  algebras, the subalgebras `so(p) + so(q)` with `p` and `q` odd, or the
  remaining special subalgebras of the exceptional algebras. Those can be
  built with `ab.via_representation` or `ab.from_subalgebra`, but they are
  not enumerated, so `ab.chains` does not pass through them.
- Subalgebras are listed by type. Two subalgebras of one type are listed
  once when they restrict the adjoint representation in the same way, which
  identifies subalgebras related by an outer automorphism. Conjugacy classes
  are not computed.
- Real forms: branching rules are for the complex algebras. Cartan
  involutions, Satake and Vogan diagrams, and the real forms of the
  subalgebras along a chain are not computed.
- Finite subgroups are out of scope here and are planned separately.
