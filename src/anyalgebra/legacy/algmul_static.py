"""Conservative, non-executing static evidence capture for ``AlgMul.wl``.

This module deliberately is not a Wolfram Language evaluator.  It tokenises
only enough of the language to locate top-level assignments and calls in a
bounded source file.  In particular, a ``ToExpression`` call is evidence that
the source *may* construct a symbol at runtime, never evidence that such a
symbol exists.  Runtime load state, registries, messages, and evaluated
property absence belong to the later runtime-capture tasks.
"""

from __future__ import annotations

import hashlib
import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from anyalgebra.core.parents import SemanticHash
from anyalgebra.legacy.models import (
    EXPECTED_PROPERTY_NAMES,
    AlgMulDefinitionRecord,
    AlgMulManifest,
    BehaviorRecord,
    DefinitionKind,
    DefinitionState,
    EvidenceOrigin,
    EvidenceOutcome,
    FactoryCallKind,
    FactoryRecord,
    LegacyPartition,
    LoadStatus,
    SourceMetadata,
    SourceSpan,
    SymbolObservation,
    SymbolRecord,
)


PINNED_ALGMUL_SHA256 = (
    "3A13C9F47092B5B9814AA08873FF4F1DE61D4635DF68FC7F3A039A7DA5B0D00D"
)
"""The audited AlgMul source identity, retained as a source-only anchor."""

MAX_SOURCE_BYTES = 8 * 1024 * 1024
_FACTORY_NAMES = ("MakeChar", "MakeTMul", "MakeAlg", "MakeProperty", "ToExpression")
_SYMBOL = re.compile(r"[A-Za-z$][A-Za-z0-9$`]*")
_DEFECT_COMMENT = re.compile(
    r"\b(?:todo|wrong|doesn'?t work|does not work|needed|"
    r"need(?:\s+to)?\s+(?:add|fix|modify|implement|support)|stub|"
    r"work[ -]?in[ -]?progress|wip|incomplete|unfinished)\b",
    re.IGNORECASE,
)

# The following catalogue is deliberately a classifier for the *legacy
# surface*, not a claim that every name has sound mathematical semantics.  It
# is grounded in ``docs/legacy/algmul-parity.md`` section 4 and uses exact
# names and narrow families before any fallback.  In particular, broad
# spelling heuristics such as ``startswith("T")`` or ``endswith("Mul")`` are
# unsafe: ``J2`` is a Jordan constructor and ``GAMul`` is geometric algebra,
# not an incomplete stub or scalar multiplication.
_REGISTRY_BASE_NAMES = frozenset(
    {
        "AddAlgName",
        "AlgToChars",
        "AlgToImagChars",
        "AlgToPureImagChars",
        "CharsToAlg",
        "e",
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
        "e6",
        "e7",
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
        "E7",
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "g",
        "g1",
        "g2",
        "g3",
        "g4",
        "g5",
        "g6",
        "g7",
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
        "G6",
        "G7",
        "h",
        "ii",
        "if1",
        "if2",
        "if3",
        "\u00cf\u0095",  # Evaluated spelling of source's \[Phi] D-basis element.
        "GetAlg",
        "ListToSymChars",
        "MakeChar",
        "MakeCharSub",
        "MakeCharSubVars",
        "MakeCharVars",
        "SetDefaultAlg",
        "StringIterateInteger",
        "StringStripInteger",
        "UniqueAlg",
        "alg",
        "CChars",
        "CsChars",
        "CxHChars",
        "HChars",
        "HsChars",
        "OChars",
        "OsChars",
        "defaultAlg",
        "defaultAlg2",
        "defaultChars",
    }
)
_ARBITRARY_STRUCTURE_NAMES = frozenset(
    {
        "MakeAlg",
        "MakeStructDot",
        "MakeSuperAlg",
        "StructMul",
    }
)
_SYMBOLIC_LIST_NAMES = frozenset(
    {
        "ListToSym",
        "MListToSym",
        "MTListToSym",
        "SymToList",
        "SymFunc",
        "SymToListOld",
        "SymToList2",
        "SymToMList",
        "SymToMTList",
        "SymToTList",
        "SymToTListOLD",
        "SymToListChars",
        "SymFoo1",
        "SymFoo2",
        "SymFoo3",
        "SymFoo4",
        "isList",
        "isSym",
        "TListToSym",
        "TLISTtoSYM",
    }
)
_ARBITRARY_ARRAY_NAMES = frozenset(
    {
        "ArrayToList",
        "DeleteAllCases",
        "ListToArray",
        "NArrayElement",
        "NSymArrayElement",
        "NTArrayElement",
        "NTSymArrayElement",
    }
)
_JORDAN_NAMES = frozenset(
    {
        "J2",
        "J2All",
        "J3",
        "J3NProof",
        "J3Proof",
        "J4",
        "JAntiAssoc",
        "FreudenthalProd",
        "JAssoc",
        "JComm",
        "JProd",
        "JordanProd",
        "IsTJ3N",
        "IsTJ3Ns",
        "TJ2",
        "TJ2Conj1",
        "TJ2Conj2",
        "TJ2ConjAll",
        "TJ3",
        "TJ3Conj1",
        "TJ3Conj2",
        "TJ3ConjAll",
        "TJ3NProof",
        "TJ3Proof",
        "TJ4",
        "TJ4Conj1",
        "TJ4Conj2",
        "TJ4ConjAll",
        "TJordanProd",
        "TJAntiAssoc",
        "TJAssoc",
        "TJComm",
        "TJProd",
    }
)
_PROPERTY_KERNEL_NAMES = frozenset(
    {
        "Alternative",
        "Assoc",
        "Comm",
        "Flexible",
        "JIdentity",
        "JAlternative",
        "JTAlternative",
        "JacobiIdentity",
        "MAlternative",
        "MAssoc",
        "MComm",
        "MFlexible",
        "MJIdentity",
        "MJacobiIdentity",
        "MPowerAssociative1",
        "MPowerAssociative2",
        "MTAlternative",
        "MTAssoc",
        "MTComm",
        "MTJIdentity",
        "MTJacobiIdentity",
        "MTPowerAssociative1",
        "MTPowerAssociative2",
        "MPowerAssociative",
        "MTPowerAssociative",
        "MTFlexible",
        "PowerAssociative",
        "PowerAssociative1",
        "PowerAssociative2",
        "TAlternative",
        "TAssoc",
        "TComm",
        "TExtAssoc",
        "TExtAssocR",
        "TFlexible",
        "TJIdentity",
        "TJacobiIdentity",
        "TPowerAssociative",
        "TPowerAssociative1",
        "TPowerAssociative2",
    }
)
_PROPERTY_WRAPPER_NAMES = frozenset(
    {
        "AMProperty1",
        "AMProperty2",
        "AMProperty3",
        "AMTProperty1",
        "AMTProperty2",
        "AMTProperty3",
        "AProperty1",
        "AProperty2",
        "AProperty3",
        "IsAAll",
        "IsAMAll",
        "IsAMTAll",
        "IsATAll",
        "IsAProperty",
        "IsNAll",
        "IsNMAll",
        "IsNMTAll",
        "IsNTAll",
        "IsNProperty",
        "ATProperty1",
        "ATProperty2",
        "ATProperty3",
        "NMProperty1",
        "NMProperty2",
        "NMProperty3",
        "NMTProperty1",
        "NMTProperty2",
        "NMTProperty3",
        "NProperty1",
        "NProperty2",
        "NProperty3",
        "NTProperty1",
        "NTProperty2",
        "NTProperty3",
    }
)
_GENERIC_ELEMENT_NAMES = frozenset(
    {
        "AElement",
        "AElementSub",
        "AMElement",
        "AMElementSub",
        "AMSymElement",
        "AMSymElementSub",
        "AMTElement",
        "AMTElementSub",
        "AMTSymElement",
        "AMTSymElementSub",
        "ASymElement",
        "ASymElementSub",
        "ATElement",
        "ATElementSub",
        "ATSymElement",
        "ATSymElementSub",
        "AVar",
        "AVarChar",
        "Avar",
        "CVar",
        "CsVar",
        "ConstantElement",
        "CxHxOVar",
        "HVar",
        "HsVar",
        "NElement",
        "NElementNew",
        "NMElement",
        "NMSymElement",
        "NMTElement",
        "NMTSymElement",
        "NSElement",
        "NSymElement",
        "NTElement",
        "NTSymElement",
        "NZeroElement",
        "OVar",
        "Ones",
        "OsVar",
        "OxOsVar",
        "TAVar",
        "TAVarChar",
        "TAvar",
        "ZeroElement",
        "Zeros",
        "VectorFix",
    }
)
_SPAN_RECOVERY_NAMES = frozenset(
    {
        "BasisDecompose",
        "BasisDecomposeRaw",
        "BasisDecomposeVariable",
        "FooToStructDot",
        "MStructConsts",
        "MStructConstsSparse",
        "MatrixToJStruct",
        "MatrixToJStructDot",
        "MatrixToLieStruct",
        "MatrixToLieStructDot",
        "MatrixToStruct",
        "MatrixToStructDot",
        "MulToStruct",
        "MulToStructDot",
        "MulToStructs",
        "MulToStructsDot",
        "StructDecompose",
        "StructToMul",
        "StructToMulTable",
    }
)
_CLIFFORD_PREFIXES = ("Cl", "Cliff", "GA", "Gr", "NAG", "Gamma")
_CLIFFORD_NAMES = frozenset(
    {
        "ChiralBasis",
        "DiracMatrices",
        "DiracToMajorana",
        "DiracToWeyl",
        "EpsLower",
        "EpsUpper",
        "FlatMetric",
        "MajoranaCharge",
        "MajoranaToDirac",
        "MajoranaToWeyl",
        "PauliMatrices",
        "SpinFund",
        "SpinorVariablesToVec",
        "VecToDiracCSpinor",
        "VecToDiracCSpinorT",
        "VecToDiracMatrices",
        "VecToDiracMatrix",
        "VecToDiracSpinor",
        "VecToDiracSpinorT",
        "VecToSpinorVariables",
        "VecToWeylCSpinor",
        "VecToWeylCSpinorT",
        "VecToWeylMatrices",
        "VecToWeylMatrix",
        "VecToWeylSpinor",
        "VecToWeylSpinorT",
        "WeylBarMatrices",
        "WeylMatrices",
        "WeylToAny",
        "WeylToDirac",
        "WeylToMajorana",
        "AnyToWeyl",
        "SymbolToTimes",
        "SymbolToTimesRule",
        "TimesToSymbol",
        "TimesToSymbolRule",
        "II",
        "XX",
        "YY",
        "ZZ",
        "iI",
        "iX",
        "iY",
        "iZ",
    }
)
_LIE_PREFIXES = ("SO", "SU", "SL", "SP", "U1", "SOn", "SUn", "SPn")
_LIE_NAMES = frozenset(
    {
        "Automorphism",
        "Automorphism1",
        "Automorphism2",
        "IsAutomorphism",
        "FindSOnMA",
        "FindSUnA",
        "FindSUnLA",
        "FindSUnMA",
        "G2Der",
        "KinSpaceDim",
        "LieProd",
        "MRootsFromChevalley",
        "MakeSOnFund",
        "MakeSUnFund",
        "MakeSPnFund",
        "TLieProd",
        "QRot",
        "Rot",
        "SAnAGens",
        "UnitaryTransformation",
    }
)
_DISPLAY_NAMES = frozenset(
    {
        "AlgebraReport",
        "GammaSummary",
        "GridList",
        "IsAAlls",
        "IsAMAlls",
        "IsAMTAlls",
        "IsAMTAllsOld",
        "IsATAlls",
        "IsATAllsOld",
        "IsNAlls",
        "IsNMAlls",
        "IsNMTAlls",
        "IsNMTAllsOld",
        "IsNMTTypes",
        "IsNTAlls",
        "IsNTAllsOld",
        "IsNTTypes",
        "MPretty",
        "MTPretty",
        "PrintProperty",
        "NMulTiming",
    }
)
_GLOBAL_OPERATOR_NAMES = frozenset(
    {
        "CenterDot",
        "CircleDot",
        "CircleTimes",
        "ClearAlls",
        "convention",
        "Diamond",
        "MakeBoxes",
        "Power",
        "SmallCircle",
        "Subscript",
        "Times",
        "Wedge",
    }
)
# These are the exact, value-less implementation names that leak from the
# pinned Wolfram 12 runtime receipt.  They are reviewed as runtime
# implementation locals, not public algebra symbols.  In particular, the
# similarly value-less basis names declared at AlgMul.wl:45-60 and the D
# element made at lines 277-278 belong in ``_REGISTRY_BASE_NAMES`` above.
# ``Z2`` is a local sign-vector parameter at lines 773-774; ``J``, ``L``, and
# ``R`` are formal associator arguments at line 1147.  No spelling heuristic
# is permitted here: a future runtime name must fail closed for review.
_RUNTIME_LEAK_NAMES = frozenset(
    {
        "a",
        "alg0",
        "alg1",
        "alg1$",
        "alg2",
        "alg2New",
        "alg2New$",
        "algebra",
        "algebra1",
        "algebra2",
        "algs",
        "algs1",
        "algsList",
        "algsList$",
        "angle",
        "aSoln",
        "aSoln$",
        "axis",
        "a$",
        "b",
        "bAnswer",
        "bAnswer$",
        "base",
        "bases",
        "base$",
        "bElem",
        "bElem$",
        "bSoln",
        "bSoln$",
        "bVars",
        "bVars$",
        "b$",
        "c",
        "char",
        "char1",
        "chars",
        "chars1",
        "chars1$",
        "chars2",
        "chars2$",
        "chars$",
        "chs",
        "chs$",
        "collapsedSign",
        "collapsedSign$",
        "counts",
        "counts$",
        "c$",
        "d",
        "data",
        "defaultChars$",
        "ds",
        "d$",
        "elem",
        "elems",
        "ending",
        "ending$",
        "expr",
        "fM",
        "foo",
        "fooAut",
        "fooElement",
        "fooList",
        "fooMul",
        "fooMul1",
        "fooMul2",
        "fooNest",
        "fooRule",
        "fooSym",
        "form",
        "H",
        "i",
        "i1",
        "i2",
        "identity",
        "identity$",
        "ids",
        "ids$",
        "imag",
        "imag$",
        "i$",
        "j",
        "J",
        "j1",
        "j2",
        "k",
        "keys",
        "L",
        "letter",
        "lhs",
        "lhs$",
        "list",
        "list$",
        "lst",
        "lst$",
        "mag",
        "mag$",
        "matrices",
        "mulBase",
        "mulBase$",
        "n",
        "name",
        "negCounts",
        "negCountsSign",
        "negCountsSign$",
        "negCounts$",
        "nest",
        "nest$",
        "newName",
        "newName$",
        "newStr",
        "newStr$",
        "ns",
        "operator",
        "p",
        "pairs",
        "pairs$",
        "param",
        "post",
        "q",
        "R",
        "reducedIndices",
        "reducedIndices$",
        "rot",
        "s",
        "scalar",
        "scalar$",
        "size",
        "str",
        "struct0",
        "struct0$",
        "struct1",
        "struct1$",
        "structDot",
        "sym",
        "sym$",
        "uniqueIndices",
        "uniqueIndices$",
        "v1",
        "v2",
        "value",
        "value$",
        "vars",
        "x",
        "y",
        "z",
        "Z2",
    }
)
_INCOMPLETE_STUB_NAMES = frozenset(
    {"GenMomenta", "GenMomentum", "MakeM", "MakeMT", "MakeT", "SYMtoTLIST"}
)
_PRODUCT_OPERATOR_PREFIXES = ("Anti", "Nest", "Nested")
_PRODUCT_OPERATOR_NAMES = frozenset(
    {
        "AssocAdd",
        "CommTable",
        "MCommTable",
        "MAntiAssoc",
        "MAntiComm",
        "MAntiCommTable",
        "MTCommTable",
        "MTAntiAssoc",
        "MTAntiComm",
        "MTAntiCommTable",
        "TCommTable",
        "TAntiComm",
        "TAntiCommTable",
        "MulSolve",
        "MulSolveR",
        "Mul",
        "MulOld",
        "MulTable",
        "TMulSolve",
        "RuleMul",
        "CombineScalarsAndIndices",
        "CombineScalarsAndIndicesRule",
        "SolveNestMul",
        "SymRuleMul",
        "MulPairs",
        "MulNewPairs",
        "MulImagPairs",
        "MulNewImagPairs",
    }
)
_CONJUGATION_NORM_NAMES = frozenset({"InnerPolarization", "Inv", "Mag"})
_CONJUGATION_NORM_TOKENS = ("Conj", "Norm", "Herm", "InnerProd", "Dvd")
_SCALAR_MULTIPLICATION_NAMES = frozenset(
    {
        "CMul",
        "CSymMul",
        "CsMul",
        "CsSymMul",
        "CxHMul",
        "CxHSymMul",
        "DMul",
        "HMul",
        "HSymMul",
        "HsMul",
        "HsSymMul",
        "InfMul",
        "LMul",
        "OMul",
        "OctonionProduct",
        "OSymMul",
        "OsMul",
        "OsSymMul",
        "SInfMul",
    }
)
_MATRIX_OVER_TENSOR_NAMES = frozenset({"MTMul", "MTMulTable", "McTMul", "cMTMul"})
_TENSOR_PRODUCT_NAMES = frozenset(
    {"CxCsMul", "MakeTMul", "TChars", "TMul", "TMulTable", "TableChars"}
)
_MATRIX_NAMES = frozenset({"McMul", "TraceReverse", "cMMul"})
_BASIS_DECLARATION_HEADS = frozenset({"e", "e1", "E1", "ii", "f1", "F1", "g1", "G1"})


class StaticCaptureError(ValueError):
    """A source file is outside the deliberately small static-parser subset."""


@dataclass(frozen=True, slots=True)
class _Statement:
    """One newline- or semicolon-delimited top-level expression and its mask."""

    text: str
    masked: str
    start_line: int
    end_line: int


@dataclass(slots=True)
class _CommentFrame:
    """One open nested Wolfram comment during defect receipt extraction."""

    start: int
    matching_child: bool = False
    child_ranges: list[tuple[int, int]] | None = None

    def add_child(self, start: int, end: int, matches: bool) -> None:
        """Remember a completed child and whether it marks a defect itself."""

        if self.child_ranges is None:
            self.child_ranges = []
        self.child_ranges.append((start, end))
        self.matching_child = self.matching_child or matches


def _hash_bytes(data: bytes) -> SemanticHash:
    return SemanticHash("sha256", hashlib.sha256(data).hexdigest())


def _masked_source(source: str) -> str:
    """Mask comments and strings while preserving every offset and newline.

    Wolfram comments nest, so regex replacement is not safe.  Strings are also
    masked so brackets and semicolons within them cannot alter statement
    boundaries.  The function performs no evaluation and has no imports beyond
    the standard library.
    """

    output = list(source)
    index = 0
    comment_depth = 0
    in_string = False
    while index < len(source):
        character = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if in_string:
            if character != "\n":
                output[index] = " "
            if character == "\\" and index + 1 < len(source):
                if source[index + 1] != "\n":
                    output[index + 1] = " "
                index += 2
                continue
            if character == '"':
                in_string = False
            index += 1
            continue
        if comment_depth:
            if character != "\n":
                output[index] = " "
            if character == "(" and following == "*":
                output[index + 1] = " "
                comment_depth += 1
                index += 2
                continue
            if character == "*" and following == ")":
                output[index + 1] = " "
                comment_depth -= 1
                index += 2
                continue
            index += 1
            continue
        if character == "(" and following == "*":
            output[index] = output[index + 1] = " "
            comment_depth = 1
            index += 2
            continue
        if character == '"':
            output[index] = " "
            in_string = True
        index += 1
    if comment_depth:
        raise StaticCaptureError("unterminated nested comment")
    if in_string:
        raise StaticCaptureError("unterminated string")
    return "".join(output)


def _defect_comment_behaviors(source: str) -> tuple[BehaviorRecord, ...]:
    """Capture only defect-marking nested comments, never their code-like text."""

    records: list[BehaviorRecord] = []
    index = 0
    comment_frames: list[_CommentFrame] = []
    in_string = False
    while index < len(source):
        character = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if in_string:
            if character == "\\" and index + 1 < len(source):
                index += 2
                continue
            if character == '"':
                in_string = False
            index += 1
            continue
        if comment_frames:
            if character == "(" and following == "*":
                comment_frames.append(_CommentFrame(index))
                index += 2
                continue
            if character == "*" and following == ")":
                frame = comment_frames.pop()
                index += 2
                comment = source[frame.start : index]
                direct_comment = comment
                for child_start, child_end in reversed(frame.child_ranges or []):
                    relative_start = child_start - frame.start
                    relative_end = child_end - frame.start
                    direct_comment = (
                        direct_comment[:relative_start]
                        + " " * (relative_end - relative_start)
                        + direct_comment[relative_end:]
                    )
                matches = _DEFECT_COMMENT.search(direct_comment) is not None
                if matches:
                    source_hash = _hash_bytes(comment.encode("utf-8"))
                    start_line = source.count("\n", 0, frame.start) + 1
                    end_line = source.count("\n", 0, index) + 1
                    ordinal = len(records) + 1
                    records.append(
                        BehaviorRecord.create(
                            f"defect-comment:{ordinal}",
                            (
                                "Static defect-marking comment; code-like text is "
                                "not parsed. "
                                f"lines {start_line}-{end_line}; comment sha256:"
                                f"{source_hash.digest}."
                            ),
                            partition=LegacyPartition.INCOMPLETE_STUBS_COMMENTS,
                            input_snapshot=source_hash,
                            origin=EvidenceOrigin.STATIC,
                            record_id=(
                                f"behavior:defect-comment:{ordinal}:{start_line}:"
                                f"{end_line}:{source_hash.digest[:16]}"
                            ),
                        )
                    )
                if comment_frames:
                    comment_frames[-1].add_child(
                        frame.start, index, matches or frame.matching_child
                    )
                continue
            index += 1
            continue
        if character == "(" and following == "*":
            comment_frames.append(_CommentFrame(index))
            index += 2
            continue
        if character == '"':
            in_string = True
        index += 1
    return tuple(records)


def _top_level_statements(source: str) -> tuple[_Statement, ...]:
    """Return bounded top-level statements, rejecting malformed brackets."""

    masked = _masked_source(source)
    stack: list[str] = []
    opening = {"[": "]", "(": ")", "{": "}"}
    start = 0
    statements: list[_Statement] = []

    def emit(end: int) -> None:
        nonlocal start
        clean = masked[start:end].rstrip()
        if clean.strip():
            prefix = len(clean) - len(clean.lstrip())
            first = start + prefix
            expression_end = start + len(clean)
            statements.append(
                _Statement(
                    # ``text`` and ``masked`` have matching offsets.  This is
                    # essential for literal call parsing after indentation,
                    # comments, or multiple expressions on one line.
                    text=source[first:expression_end],
                    masked=clean[prefix:],
                    start_line=source.count("\n", 0, first) + 1,
                    end_line=source.count("\n", 0, expression_end) + 1,
                )
            )
        start = end

    for index, character in enumerate(masked):
        if character in opening:
            stack.append(opening[character])
        elif character in "]) }".replace(" ", ""):
            if not stack or stack.pop() != character:
                raise StaticCaptureError("malformed bracket structure")
        elif character == ";" and not stack:
            emit(index + 1)
        elif character == "\n" and not stack:
            current = masked[start:index].rstrip()
            if current and not _continues_at_newline(current, source[start:index]):
                emit(index)
            elif not current:
                start = index + 1
    if stack:
        raise StaticCaptureError("unterminated bracket structure")
    emit(len(source))
    return tuple(statements)


def _continues_at_newline(masked: str, raw: str) -> bool:
    """Recognise only explicit top-level continuations after a physical line."""

    if not masked.endswith((":=", "=", "+", "-", "*", "/", ",", "\\")):
        return False
    # Strings are blanked in ``masked`` but are real right-hand expressions.
    # Looking at the last raw character prevents a complete ``f := \"x\"``
    # line from being incorrectly joined to the next definition.
    return raw.rstrip().endswith((":=", "=", "+", "-", "*", "/", ",", "\\"))


def _assignment_operator(masked: str) -> int | None:
    """Find the first conservative Wolfram Set or SetDelayed operator."""

    stack: list[str] = []
    opening = {"[": "]", "(": ")", "{": "}"}
    for index, character in enumerate(masked):
        if character in opening:
            stack.append(opening[character])
            continue
        if character in "]) }".replace(" ", ""):
            if stack:
                stack.pop()
            continue
        if stack:
            continue
        if character != "=":
            continue
        before = masked[index - 1] if index else ""
        after = masked[index + 1] if index + 1 < len(masked) else ""
        if (before and before in "<>=!/-") or (after and after in "=>"):
            continue
        return index
    return None


def _assignment_name(masked: str) -> tuple[str, DefinitionKind] | None:
    """Extract a named top-level assignment without interpreting its RHS."""

    operator = _assignment_operator(masked)
    if operator is None:
        return None
    if re.search(r"\^:?=", masked[: operator + 1]) is not None:
        raise StaticCaptureError("unsupported UpSet assignment")
    lhs = masked[:operator]
    names = tuple(_SYMBOL.finditer(lhs))
    if not names:
        return None
    # ``tag /: head[...] := ...`` is an UpValue attached to ``tag``.  It is
    # not a DownValue on ``head``.  Ordinary Part/DownValue assignments such
    # as alg["chars"][x_] keep their first head.
    if "/:" in lhs:
        return names[0].group(0).split("`")[-1], DefinitionKind.UP_VALUE
    name_match = names[0]
    bracket_groups = _top_level_bracket_groups(lhs)
    kind = (
        DefinitionKind.SUB_VALUE
        if bracket_groups >= 2
        else DefinitionKind.DOWN_VALUE
        if bracket_groups == 1
        else DefinitionKind.OWN_VALUE
    )
    return name_match.group(0).split("`")[-1], kind


def _top_level_bracket_groups(lhs: str) -> int:
    """Count consecutive top-level application groups without parsing the RHS."""

    depth = 0
    groups = 0
    for character in lhs:
        if character == "[":
            if depth == 0:
                groups += 1
            depth += 1
        elif character == "]" and depth:
            depth -= 1
    return groups


def _usage_message_name(masked: str) -> str | None:
    """Return a top-level ``name::usage = ...`` documentation assignment.

    ``MessageName[name, \"usage\"]`` is an own value of the message name, not
    an OwnValue of ``name``.  It must never turn a function with DownValues into
    a fictitious mixed definition.
    """

    operator = _assignment_operator(masked)
    if operator is None:
        return None
    lhs = masked[:operator].strip()
    match = re.fullmatch(r"([A-Za-z$][A-Za-z0-9$`]*)::usage", lhs)
    return None if match is None else match.group(1).split("`")[-1]


_RUNTIME_PARTITION_BY_NAME: dict[str, LegacyPartition] = {
    **{n: LegacyPartition.GLOBAL_ALIASES_OPERATORS for n in _RUNTIME_LEAK_NAMES},
    **{n: LegacyPartition.ARBITRARY_ARRAYS for n in ["ArrayToList", "ListToArray"]},
    **{
        n: LegacyPartition.ARBITRARY_STRUCTURES
        for n in ["MakeAlg", "MakeStructDot", "MakeSuperAlg", "StructMul"]
    },
    **{
        n: LegacyPartition.CLIFFORD_GEOMETRIC_GRASSMANN
        for n in [
            "GAMul",
            "GAReduceIndices",
            "GAReduceIndicesRule",
            "GrMul",
            "GrReduceIndices",
            "GrReduceIndicesRule",
            "NAGMul",
            "NAGReduceIndices",
            "NAGReduceIndicesRule",
            "SymbolToTimes",
            "SymbolToTimesRule",
            "TimesToSymbol",
            "TimesToSymbolRule",
        ]
    },
    **{
        n: LegacyPartition.CONJUGATIONS_NORMS
        for n in [
            "AntiHermCConj",
            "AntiHermCHConj",
            "AntiHermCHOConj",
            "AntiHermCOConj",
            "AntiHermHConj",
            "AntiHermHOConj",
            "AntiHermOConj",
            "CConj",
            "CHConj",
            "CHOConj",
            "COConj",
            "Conj",
            "CxHxOConj",
            "CxOConj",
            "DoubleNorm2",
            "Dvd",
            "DvdL",
            "DvdR",
            "HConj",
            "HOConj",
            "HermCConj",
            "HermCHConj",
            "HermCHOConj",
            "HermCOConj",
            "HermConj",
            "HermHConj",
            "HermHOConj",
            "HermMul",
            "HermOConj",
            "HermTConj",
            "HermTConj1",
            "HermTConj2",
            "HermTConjAll",
            "HxOConj",
            "InnerPolarization",
            "InnerProd",
            "Inv",
            "LConjNorm2",
            "LRConjNorm2",
            "LRConjNorm2Comm",
            "MCConj",
            "MCHConj",
            "MCHOConj",
            "MCOConj",
            "MConj",
            "MHConj",
            "MHOConj",
            "MInnerProd",
            "MOConj",
            "MSymConj",
            "MTConj",
            "MTConj1",
            "MTConj2",
            "MTConjAll",
            "Mag",
            "NTLRConjNorm2CommTable",
            "NestMulDConj",
            "Norm2",
            "Normize",
            "OConj",
            "RConjNorm2",
            "SymConj",
            "SymDvd",
            "TConj",
            "TConj1",
            "TConj2",
            "TConjAll",
            "TDoubleNorm2",
            "TLConjNorm2",
            "TLRConjNorm2",
            "TLRConjNorm2Comm",
            "TRConjNorm2",
            "TSymConj",
            "TSymConj1",
            "TSymConj2",
            "TSymConjAll",
            "fooConj",
            "fooConjs",
            "fooDvd",
            "fooNorm",
        ]
    },
    **{n: LegacyPartition.DISPLAY_REPORT_TABLES for n in ["GridList", "NMulTiming"]},
    **{
        n: LegacyPartition.GENERIC_ELEMENTS
        for n in ["AElement", "AElementSub", "ATElement", "NTElement"]
    },
    "SYMtoTLIST": LegacyPartition.INCOMPLETE_STUBS_COMMENTS,
    **{n: LegacyPartition.JORDAN_MATRICES for n in ["JProd", "TJordanProd"]},
    **{
        n: LegacyPartition.LIE_GROUP_EXPERIMENTS
        for n in [
            "Automorphism",
            "Automorphism1",
            "Automorphism2",
            "G2Der",
            "IsAutomorphism",
            "MakeSOnFund",
            "MakeSPnFund",
            "MakeSUnFund",
            "QRot",
            "Rot",
            "SOnFund",
            "SUnFund",
        ]
    },
    **{
        n: LegacyPartition.MATRICES
        for n in [
            "MDet2",
            "MMul",
            "MMulTable",
            "MTMul",
            "MTMulTable",
            "MakeTChars",
            "McMul",
            "McTMul",
            "MulTableToStruct",
            "cMMul",
            "cMTMul",
        ]
    },
    **{
        n: LegacyPartition.PRODUCT_OPERATORS
        for n in [
            "AntiAssoc",
            "AntiComm",
            "AntiCommTable",
            "AssocAdd",
            "CombineScalarsAndIndices",
            "CombineScalarsAndIndicesRule",
            "CommTable",
            "MAntiComm",
            "MAntiCommTable",
            "MCommTable",
            "MTAntiComm",
            "MTAntiCommTable",
            "MTCommTable",
            "Mul",
            "MulImagPairs",
            "MulNewImagPairs",
            "MulNewPairs",
            "MulOld",
            "MulPairs",
            "MulSolve",
            "MulSolveR",
            "MulTable",
            "Nest2Mul",
            "NestAntiCommMul",
            "NestAntiCommMulD",
            "NestAntiCommMulR",
            "NestCommMul",
            "NestCommMulD",
            "NestCommMulR",
            "NestCommTrans",
            "NestCommTransD",
            "NestCommTransR",
            "NestMul",
            "NestMulD",
            "NestMulDF",
            "NestMulR",
            "NestMulRF",
            "NestedMul",
            "NestedMulD",
            "NestedMulR",
            "RuleMul",
            "SolveNestMul",
            "SymRuleMul",
            "TAntiComm",
            "TAntiCommTable",
            "TCommTable",
            "TMulSolve",
        ]
    },
    **{
        n: LegacyPartition.PROPERTY_KERNELS
        for n in [
            "Assoc",
            "Comm",
            "MComm",
            "MTComm",
            "TAssoc",
            "TComm",
            "TExtAssoc",
            "TExtAssocR",
        ]
    },
    **{
        n: LegacyPartition.REGISTRY_BASES
        for n in [
            "AddAlgName",
            "AlgToChars",
            "AlgToImagChars",
            "AlgToPureImagChars",
            "CChars",
            "CharsToAlg",
            "CsChars",
            "CxHChars",
            "E1",
            "E2",
            "E3",
            "E4",
            "E5",
            "E6",
            "E7",
            "F1",
            "F2",
            "F3",
            "F4",
            "F5",
            "F6",
            "F7",
            "G1",
            "G2",
            "G3",
            "G4",
            "G5",
            "G6",
            "G7",
            "GetAlg",
            "HChars",
            "HsChars",
            "ListToSymChars",
            "MakeChar",
            "MakeCharSub",
            "MakeCharSubVars",
            "MakeCharVars",
            "OChars",
            "OsChars",
            "SetDefaultAlg",
            "StringIterateInteger",
            "StringStripInteger",
            "UniqueAlg",
            "alg",
            "defaultAlg",
            "defaultAlg2",
            "defaultChars",
            "e",
            "e1",
            "e2",
            "e3",
            "e4",
            "e5",
            "e6",
            "e7",
            "f1",
            "f2",
            "f3",
            "f4",
            "f5",
            "f6",
            "f7",
            "g",
            "g1",
            "g2",
            "g3",
            "g4",
            "g5",
            "g6",
            "g7",
            "h",
            "if1",
            "if2",
            "if3",
            "ii",
        ]
    },
    "\u00cf\u0095": LegacyPartition.REGISTRY_BASES,
    **{
        n: LegacyPartition.SCALAR_COMPOSITION_PRODUCTS
        for n in [
            "CMul",
            "CSymMul",
            "CsMul",
            "CsSymMul",
            "CxHMul",
            "CxHSymMul",
            "DMul",
            "HMul",
            "HSymMul",
            "HsMul",
            "HsSymMul",
            "InfMul",
            "LMul",
            "OMul",
            "OSymMul",
            "OctonionProduct",
            "OsMul",
            "OsSymMul",
            "SInfMul",
        ]
    },
    **{
        n: LegacyPartition.SPAN_STRUCTURE_RECOVERY
        for n in [
            "FooToStructDot",
            "MStructConsts",
            "MStructConstsSparse",
            "MatrixToJStruct",
            "MatrixToJStructDot",
            "MatrixToLieStruct",
            "MatrixToLieStructDot",
            "MatrixToStruct",
            "MatrixToStructDot",
            "MulToStruct",
            "MulToStructDot",
            "MulToStructs",
            "MulToStructsDot",
            "StructDecompose",
            "StructToMul",
            "StructToMulTable",
        ]
    },
    **{
        n: LegacyPartition.SYMBOLIC_LIST_CONVERSION
        for n in [
            "ListToSym",
            "MListToSym",
            "MTListToSym",
            "SymFoo1",
            "SymFoo2",
            "SymFoo3",
            "SymFoo4",
            "SymFunc",
            "SymToList",
            "SymToList2",
            "SymToListChars",
            "SymToListOld",
            "SymToMList",
            "SymToMTList",
            "SymToTList",
            "SymToTListOLD",
            "TLISTtoSYM",
            "TListToSym",
            "isList",
            "isSym",
        ]
    },
    **{
        n: LegacyPartition.TENSOR_PRODUCTS
        for n in ["CxCsMul", "MakeTMul", "TChars", "TMul", "TMulTable", "TableChars"]
    },
}


def _partition(
    name: str, *, strict: bool = False, runtime_surface: bool = False
) -> LegacyPartition:
    """Classify a named AlgMul surface using the section-4 catalogue.

    Exact tables take precedence over families so an overloaded spelling stays
    reviewable.  The terminal incomplete classification is intentionally
    narrow: it represents an explicitly unfinished legacy name, never a
    catch-all for established mathematical families.  ``runtime_surface``
    requires exact catalogue membership and fails closed on a new evaluated
    name, including a new runtime implementation local.
    """

    if runtime_surface:
        partition = _RUNTIME_PARTITION_BY_NAME.get(name)
        if partition is None:
            raise StaticCaptureError(f"unreviewed evaluated AlgMul name: {name}")
        return partition
    if name in EXPECTED_PROPERTY_NAMES or name == "MakeProperty":
        return LegacyPartition.GENERATED_PROPERTY_API
    if name in _DISPLAY_NAMES or name.endswith(("Report", "Summary", "Pretty")):
        return LegacyPartition.DISPLAY_REPORT_TABLES
    if name in _GLOBAL_OPERATOR_NAMES:
        return LegacyPartition.GLOBAL_ALIASES_OPERATORS
    if name in _REGISTRY_BASE_NAMES:
        return LegacyPartition.REGISTRY_BASES
    if name in _ARBITRARY_STRUCTURE_NAMES:
        return LegacyPartition.ARBITRARY_STRUCTURES
    if name in _SYMBOLIC_LIST_NAMES:
        return LegacyPartition.SYMBOLIC_LIST_CONVERSION
    if name in _ARBITRARY_ARRAY_NAMES:
        return LegacyPartition.ARBITRARY_ARRAYS
    if name in _JORDAN_NAMES:
        return LegacyPartition.JORDAN_MATRICES
    if name in _PROPERTY_KERNEL_NAMES:
        return LegacyPartition.PROPERTY_KERNELS
    if name in _PROPERTY_WRAPPER_NAMES:
        return LegacyPartition.PROPERTY_WRAPPERS
    if name in _GENERIC_ELEMENT_NAMES:
        return LegacyPartition.GENERIC_ELEMENTS
    if name in _SPAN_RECOVERY_NAMES:
        return LegacyPartition.SPAN_STRUCTURE_RECOVERY
    if name in _CONJUGATION_NORM_NAMES or any(
        token in name for token in _CONJUGATION_NORM_TOKENS
    ):
        return LegacyPartition.CONJUGATIONS_NORMS
    if name in _CLIFFORD_NAMES or name.startswith(_CLIFFORD_PREFIXES):
        return LegacyPartition.CLIFFORD_GEOMETRIC_GRASSMANN
    if name in _LIE_NAMES or name.startswith(_LIE_PREFIXES):
        return LegacyPartition.LIE_GROUP_EXPERIMENTS
    if name in _INCOMPLETE_STUB_NAMES:
        return LegacyPartition.INCOMPLETE_STUBS_COMMENTS
    if name in _PRODUCT_OPERATOR_NAMES or name.startswith(_PRODUCT_OPERATOR_PREFIXES):
        return LegacyPartition.PRODUCT_OPERATORS
    if name in _SCALAR_MULTIPLICATION_NAMES:
        return LegacyPartition.SCALAR_COMPOSITION_PRODUCTS
    if name in _MATRIX_OVER_TENSOR_NAMES:
        return LegacyPartition.MATRICES
    if name in _TENSOR_PRODUCT_NAMES:
        return LegacyPartition.TENSOR_PRODUCTS
    if name.startswith("M") or "Matrix" in name or name in _MATRIX_NAMES:
        return LegacyPartition.MATRICES
    if strict:
        raise StaticCaptureError(f"unreviewed AlgMul catalogue name: {name}")
    return LegacyPartition.INCOMPLETE_STUBS_COMMENTS


def _executable_behavior(statement: _Statement, ordinal: int) -> BehaviorRecord:
    """Record one non-assignment top-level expression as static evidence."""

    head_match = _SYMBOL.search(statement.masked)
    head = "anonymous" if head_match is None else head_match.group(0).split("`")[-1]
    targets = _top_level_call_targets(statement.masked, head)
    source_hash = _hash_bytes(statement.text.encode("utf-8"))
    target_id = "none" if not targets else "-".join(targets)
    return BehaviorRecord.create(
        f"static-expression:{head}:{target_id}:{ordinal}",
        (
            "Static top-level executable expression; "
            f"head={head}; targets={','.join(targets) or 'none'}; "
            f"lines {statement.start_line}-{statement.end_line}; "
            f"expression sha256:{source_hash.digest}."
        ),
        partition=_expression_partition(statement, head),
        input_snapshot=source_hash,
        origin=EvidenceOrigin.STATIC,
        record_id=(
            f"behavior:static-expression:{head}:{target_id}:{ordinal}:"
            f"{statement.start_line}:{statement.end_line}:{source_hash.digest[:16]}"
        ),
    )


def _expression_partition(statement: _Statement, head: str) -> LegacyPartition:
    """Classify a source-level effect from its full bounded statement receipt."""

    # These source-anchored forms are not function calls and their first token
    # does not identify their operational role.  Keep the tests pinned to the
    # captured line anchors so a source revision cannot silently retain them.
    if head == "Do" and 'alg["charsList"]' in statement.text:
        return LegacyPartition.REGISTRY_BASES
    if head in _BASIS_DECLARATION_HEADS and statement.masked.lstrip().startswith("{"):
        return LegacyPartition.REGISTRY_BASES
    if "MatrixForm" in statement.text:
        return LegacyPartition.DISPLAY_REPORT_TABLES
    if head == "MakeTChars":
        return LegacyPartition.INCOMPLETE_STUBS_COMMENTS

    if head in {"SetAttributes", "Protect", "Unprotect"}:
        return LegacyPartition.GLOBAL_ALIASES_OPERATORS
    if head == "Print":
        return LegacyPartition.DISPLAY_REPORT_TABLES
    if head in {"Clear", "ClearAll", "BeginPackage", "EndPackage", "Begin", "End"}:
        return LegacyPartition.GLOBAL_ALIASES_OPERATORS
    return _partition(head)


def _top_level_call_targets(masked: str, head: str) -> tuple[str, ...]:
    """Extract direct target symbols from one top-level effect call."""

    if head == "anonymous":
        return ()
    match = re.match(rf"\s*{re.escape(head)}\s*\[", masked)
    if match is None:
        return ()
    opening = {"[": "]", "(": ")", "{": "}"}
    stack = ["]"]
    arguments: list[str] = []
    argument_start = match.end()
    for index, character in enumerate(masked[match.end() :], start=match.end()):
        if character in opening:
            stack.append(opening[character])
            continue
        if stack and character == stack[-1]:
            stack.pop()
            if not stack:
                arguments.append(masked[argument_start:index])
                break
            continue
        if character == "," and len(stack) == 1:
            arguments.append(masked[argument_start:index])
            argument_start = index + 1
    candidates = arguments[:1] if head == "SetAttributes" else arguments
    return tuple(
        symbol.group(0).split("`")[-1]
        for argument in candidates
        for symbol in _SYMBOL.finditer(argument)
    )


def _call_sites(statements: Iterable[_Statement], name: str) -> tuple[str, ...]:
    """Return deterministic source-only call-site identifiers for one head."""

    matcher = re.compile(rf"(?<![A-Za-z0-9$`]){re.escape(name)}\s*\[")
    found: list[str] = []
    for statement in statements:
        # A factory's own left-hand definition is source evidence, but not a
        # separate invocation.  Calls inside its body remain useful dynamic
        # construction evidence and are retained.
        operator = _assignment_operator(statement.masked)
        for occurrence, match in enumerate(matcher.finditer(statement.masked), start=1):
            # The left-hand head of a definition is a definition site, not an
            # invocation.  Calls on its right-hand side remain captured.
            if operator is not None and match.start() < operator:
                continue
            digest = _hash_bytes(statement.text.encode("utf-8")).digest[:16]
            found.append(
                f"callsite:{name}:{statement.start_line}:{statement.end_line}:{occurrence}:{digest}"
            )
    return tuple(found)


def _property_surface_requested(statements: Iterable[_Statement]) -> bool:
    """Recognise the audited eight literal property-factory requests only.

    A file that merely defines a function named ``MakeProperty`` does not
    warrant 128 invented intended names.  The source must contain the eight
    concrete family requests that explain that specific intended surface.
    """

    literal_matcher = re.compile(
        r'MakeProperty\s*\[\s*"([A-Za-z0-9]+)"\s*,\s*([0-9]+)\s*,\s*'
        r'"([A-Za-z0-9]+)"\s*\]\s*;?\s*'
    )
    requested: list[tuple[str, int, str]] = []
    for statement in statements:
        if _assignment_name(statement.masked) is not None:
            continue
        literal = literal_matcher.fullmatch(statement.text)
        if literal is not None:
            requested.append(
                (literal.group(1), int(literal.group(2)), literal.group(3))
            )
    return tuple(requested) == (
        ("Commutative", 2, "Comm"),
        ("Associative", 3, "Assoc"),
        ("Alternative", 2, "Alternative"),
        ("Flexible", 2, "Flexible"),
        ("PowerAssociative1", 1, "PowerAssociative1"),
        ("PowerAssociative2", 1, "PowerAssociative2"),
        ("JIdentity", 2, "JIdentity"),
        ("Jacobi", 3, "JacobiIdentity"),
    )


def _makeproperty_factory_definition(
    assignments: Iterable[tuple[str, DefinitionKind, _Statement]],
) -> bool:
    """Require a literal three-argument DownValue before inferring its family."""

    for name, kind, statement in assignments:
        if name != "MakeProperty" or kind is not DefinitionKind.DOWN_VALUE:
            continue
        operator = _assignment_operator(statement.masked)
        if operator is None:
            continue
        lhs = statement.masked[:operator]
        if _top_level_call_arity(lhs, "MakeProperty") == 3:
            return True
    return False


def _top_level_call_arity(lhs: str, head: str) -> int | None:
    """Return the first application arity for one literal head on a held LHS."""

    match = re.match(rf"\s*{re.escape(head)}\s*\[", lhs)
    if match is None:
        return None
    depth = 1
    commas = 0
    content = False
    for character in lhs[match.end() :]:
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return commas + 1 if content else 0
        elif depth == 1:
            if character == ",":
                commas += 1
            elif not character.isspace():
                content = True
    return None


def _static_capture(path: Path, source_bytes: bytes) -> AlgMulManifest:
    """Build a source-only manifest from already bounded bytes."""

    try:
        source = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StaticCaptureError("source must be valid UTF-8") from error
    statements = _top_level_statements(source)
    source_hash = _hash_bytes(source_bytes)
    observations = (
        SymbolObservation.create(EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT),
    )
    symbols_by_name: dict[str, SymbolRecord] = {}
    assignments: list[tuple[str, DefinitionKind, _Statement]] = []
    usage_messages: list[tuple[str, _Statement]] = []
    assignment_statements: set[_Statement] = set()
    usage_statements: set[_Statement] = set()
    counts_by_name: dict[str, dict[DefinitionKind, int]] = {}
    for statement in statements:
        usage_name = _usage_message_name(statement.masked)
        if usage_name is not None:
            usage_messages.append((usage_name, statement))
            usage_statements.add(statement)
            continue
        assignment = _assignment_name(statement.masked)
        if assignment is None:
            continue
        name, kind = assignment
        assignments.append((name, kind, statement))
        assignment_statements.add(statement)
        counts_by_name.setdefault(name, {}).setdefault(kind, 0)
        counts_by_name[name][kind] += 1
    for name in counts_by_name:
        symbols_by_name[name] = SymbolRecord.create(
            name, _partition(name), observations
        )
    function_definition_names = {
        name for name, kind, _ in assignments if kind is DefinitionKind.DOWN_VALUE
    }
    definitions: list[AlgMulDefinitionRecord] = []
    occurrence_by_name: dict[str, int] = {}
    for name, kind, statement in assignments:
        occurrence_by_name[name] = occurrence_by_name.get(name, 0) + 1
        symbol = symbols_by_name[name]
        definitions.append(
            AlgMulDefinitionRecord.create(
                symbol.record_id,
                EvidenceOrigin.STATIC,
                DefinitionState.PRESENT_DEFINED,
                kind,
                downvalue_count=1 if kind is DefinitionKind.DOWN_VALUE else 0,
                ownvalue_count=1 if kind is DefinitionKind.OWN_VALUE else 0,
                upvalue_count=1 if kind is DefinitionKind.UP_VALUE else 0,
                subvalue_count=1 if kind is DefinitionKind.SUB_VALUE else 0,
                source_span=SourceSpan.create(statement.start_line, statement.end_line),
                definition_hash=_hash_bytes(statement.text.encode("utf-8")),
                record_id=f"definition:{symbol.record_id}:static:{occurrence_by_name[name]}",
            )
        )
    # Intended names occur only through dynamic construction.  They are not
    # literal source definitions and are intentionally recorded as neutral
    # INTENDED shells, without static or evaluated definition evidence.
    property_surface_requested = _makeproperty_factory_definition(
        assignments
    ) and _property_surface_requested(statements)
    if property_surface_requested:
        for name in EXPECTED_PROPERTY_NAMES:
            symbols_by_name.setdefault(
                name,
                SymbolRecord.create(
                    name,
                    LegacyPartition.GENERATED_PROPERTY_API,
                    (
                        SymbolObservation.create(
                            EvidenceOrigin.INTENDED, EvidenceOutcome.PRESENT
                        ),
                    ),
                ),
            )
    factory_specs = (
        ("MakeChar", LegacyPartition.REGISTRY_BASES, (), FactoryCallKind.GENERATES),
        ("MakeTMul", LegacyPartition.TENSOR_PRODUCTS, (), FactoryCallKind.GENERATES),
        (
            "MakeAlg",
            LegacyPartition.ARBITRARY_STRUCTURES,
            (),
            FactoryCallKind.GENERATES,
        ),
        (
            "MakeProperty",
            LegacyPartition.GENERATED_PROPERTY_API,
            EXPECTED_PROPERTY_NAMES if property_surface_requested else (),
            FactoryCallKind.GENERATES,
        ),
        ("ToExpression", LegacyPartition.REGISTRY_BASES, (), FactoryCallKind.GENERATES),
    )
    call_sites_by_name = {
        name: _call_sites(statements, name) for name in _FACTORY_NAMES
    }
    factories_list: list[FactoryRecord] = []
    for name, partition, requested_names, call_kind in factory_specs:
        call_relationships = call_sites_by_name[name]
        if name == "ToExpression":
            if not call_relationships:
                continue
        elif name not in function_definition_names:
            continue
        if name == "MakeProperty" and not property_surface_requested:
            continue
        factories_list.append(
            FactoryRecord.create(
                name,
                (
                    SymbolObservation.create(
                        EvidenceOrigin.STATIC, EvidenceOutcome.PRESENT
                    ),
                    *(
                        (
                            SymbolObservation.create(
                                EvidenceOrigin.INTENDED, EvidenceOutcome.PRESENT
                            ),
                        )
                        if name == "MakeProperty"
                        else ()
                    ),
                ),
                requested_names=requested_names,
                partition=partition,
                call_kind=call_kind,
                call_relationships=call_relationships,
            )
        )
    factories = tuple(factories_list)
    callsite_partitions = {
        "MakeChar": LegacyPartition.REGISTRY_BASES,
        "MakeTMul": LegacyPartition.TENSOR_PRODUCTS,
        "MakeAlg": LegacyPartition.ARBITRARY_STRUCTURES,
        "MakeProperty": LegacyPartition.GENERATED_PROPERTY_API,
        "ToExpression": LegacyPartition.REGISTRY_BASES,
    }
    dynamic_behaviors = tuple(
        BehaviorRecord.create(
            f"static-callsite:{name}",
            (
                "Static call-site evidence only; runtime outputs are intentionally "
                f"not inferred. count={len(call_sites)}; callsite-list sha256:"
                f"{hashlib.sha256('|'.join(call_sites).encode('utf-8')).hexdigest()}."
            ),
            partition=callsite_partitions[name],
            input_snapshot=_hash_bytes("|".join(call_sites).encode("utf-8")),
            origin=EvidenceOrigin.STATIC,
            record_id=f"behavior:static-callsite:{name}",
        )
        for name, call_sites in call_sites_by_name.items()
        if call_sites
    )
    usage_behaviors = tuple(
        BehaviorRecord.create(
            f"usage-message:{name}:{statement.start_line}",
            (
                "Static MessageName usage assignment (not a function OwnValue); "
                f"lines {statement.start_line}-{statement.end_line}; "
                f"slice sha256:{_hash_bytes(statement.text.encode('utf-8')).digest}."
            ),
            partition=_partition(name),
            input_snapshot=_hash_bytes(statement.text.encode("utf-8")),
            origin=EvidenceOrigin.STATIC,
            record_id=f"behavior:usage-message:{name}:{statement.start_line}",
        )
        for name, statement in usage_messages
    )
    executable_behaviors = tuple(
        _executable_behavior(statement, ordinal)
        for ordinal, statement in enumerate(statements, start=1)
        if statement not in assignment_statements and statement not in usage_statements
    )
    return AlgMulManifest.create(
        manifest_id=f"algmul-static:{source_hash.digest}",
        source=SourceMetadata.create(
            str(path),
            source_hash,
            source_bytes=len(source_bytes),
            load_status=LoadStatus.STATIC_ONLY,
        ),
        symbols=tuple(symbols_by_name.values()),
        definitions=tuple(definitions),
        factories=factories,
        registries=(),
        behaviors=(
            *dynamic_behaviors,
            *usage_behaviors,
            *executable_behaviors,
            *_defect_comment_behaviors(source),
        ),
        messages=(),
        dispositions=(),
    )


def capture_static(source_path: str | Path) -> AlgMulManifest:
    """Capture source-only AlgMul evidence without executing Wolfram code.

    The input must be a regular UTF-8 file no larger than :data:`MAX_SOURCE_BYTES`.
    The pinned hash is an exported comparison constant, not a permission to add
    evaluated evidence: both pinned and arbitrary files remain ``STATIC_ONLY``.
    """

    try:
        path = Path(source_path).resolve(strict=True)
        metadata = path.stat()
    except (OSError, RuntimeError, TypeError) as error:
        raise StaticCaptureError("source must be a regular file") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise StaticCaptureError("source must be a regular file")
    try:
        with path.open("rb") as source_file:
            source_bytes = source_file.read(MAX_SOURCE_BYTES + 1)
    except OSError as error:
        raise StaticCaptureError("source could not be read") from error
    if len(source_bytes) > MAX_SOURCE_BYTES:
        raise StaticCaptureError("source exceeds bounded static capture limit")
    return _static_capture(path, source_bytes)


def capture_pinned_static(source_path: str | Path) -> AlgMulManifest:
    """Capture only the audited AlgMul source, rejecting a hash mismatch.

    This helper is a source-identity gate, not a runtime merge.  Its successful
    result remains strictly ``STATIC_ONLY`` and contains no kernel-derived data.
    """

    manifest = capture_static(source_path)
    if manifest.source.content_hash.digest.upper() != PINNED_ALGMUL_SHA256:
        raise StaticCaptureError("source hash does not match pinned AlgMul receipt")
    unreviewed = sorted(
        item.name
        for item in manifest.symbols
        if item.partition is LegacyPartition.INCOMPLETE_STUBS_COMMENTS
        and item.name not in _INCOMPLETE_STUB_NAMES
    )
    if unreviewed:
        raise StaticCaptureError(
            "pinned source has unreviewed catalogue names: " + ", ".join(unreviewed)
        )
    return manifest


class AlgMulAudit:
    """Namespace-compatible facade for the non-executing static capture API."""

    @staticmethod
    def capture_static(source_path: str | Path) -> AlgMulManifest:
        """Delegate to :func:`capture_static` without adding runtime behavior."""

        return capture_static(source_path)
