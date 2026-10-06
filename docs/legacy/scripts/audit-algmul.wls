(* Clean-kernel evaluated surface audit for AlgMul.wl. *)

source = Environment["ANYALGEBRA_ALGMUL_SOURCE"];
If[!StringQ[source] || StringLength[source] == 0,
  Print["Set ANYALGEBRA_ALGMUL_SOURCE to the external AlgMul.wl path."];
  Exit[2]
];
repoRoot = Nest[DirectoryName, $InputFileName, 4];
output = FileNameJoin[{repoRoot, "docs", "legacy", "generated", "algmul-evaluated-surface.json"}];

beforeGlobalNames = Names["Global`*"];
beforePackageNames = Names["AlgMul`*"];
initialContext = $Context;
initialContextPath = $ContextPath;
started = DateString[{"ISODate", "T", "Time", "Z"}];
loadResult = TimeConstrained[Quiet[Get[source]], 180, $Aborted];
Global`contextAfterLoad = $Context;
Global`contextPathAfterLoad = $ContextPath;
Global`packageNamesImmediatelyAfterLoad = Names["AlgMul`*"];
$Context = initialContext;
$ContextPath = initialContextPath;
finished = DateString[{"ISODate", "T", "Time", "Z"}];
afterGlobalNames = Names["Global`*"];
afterPackageNames = Global`packageNamesImmediatelyAfterLoad;
newGlobalNames = Sort[Complement[afterGlobalNames, beforeGlobalNames]];
newPackageNames = Sort[Complement[afterPackageNames, beforePackageNames]];
newNames = Join[newPackageNames, newGlobalNames];

symbolSummary[name_String] := ToExpression[
  name,
  InputForm,
  Function[s,
    <|
      "name" -> name,
      "attributes" -> (ToString[#, InputForm] & /@ Attributes[s]),
      "ownValueCount" -> Length[OwnValues[s]],
      "downValueCount" -> Length[DownValues[s]],
      "upValueCount" -> Length[UpValues[s]],
      "subValueCount" -> Length[SubValues[s]]
    |>,
    HoldAll
  ]
];

summaries = symbolSummary /@ newNames;
definedNames = Lookup[
  Select[summaries, Total[Lookup[#, {"ownValueCount", "downValueCount", "upValueCount", "subValueCount"}]] > 0 &],
  "name"
];

propertyNames = {
  "Commutative", "Associative", "Alternative", "Flexible",
  "PowerAssociative1", "PowerAssociative2", "JIdentity", "Jacobi"
};
propertyPrefixes = {
  "A", "AT", "AM", "AMT", "IsA", "IsAT", "IsAM", "IsAMT",
  "N", "NT", "NM", "NMT", "IsN", "IsNT", "IsNM", "IsNMT"
};
expectedPropertySymbols = Flatten[Outer[StringJoin, propertyPrefixes, propertyNames]];
expectedPropertyMatches = Association@Table[
  propertyName -> DeleteDuplicates[Join[
    Names["AlgMul`" <> propertyName],
    Names["Global`" <> propertyName],
    Names["*`" <> propertyName]
  ]],
  {propertyName, expectedPropertySymbols}
];
propertyRecords = (<|
    "name" -> #,
    "matches" -> Lookup[expectedPropertyMatches, #, {}],
    "present" -> Length[Lookup[expectedPropertyMatches, #, {}]] > 0,
    "defined" -> AnyTrue[Lookup[expectedPropertyMatches, #, {}],
      Function[n, With[{record = symbolSummary[n]},
        Total[Lookup[record, {"ownValueCount", "downValueCount", "upValueCount", "subValueCount"}]] > 0
      ]]]
  |> &) /@ expectedPropertySymbols;

propertyFactoryExerciseResult = Quiet[Check[
  AlgMul`MakeProperty["AuditGenerated", 2, "Comm"],
  $Failed
]];
propertyFactoryExerciseNames = DeleteDuplicates[Join[
  Names["AlgMul`*AuditGenerated*"],
  Names["Global`*AuditGenerated*"],
  Names["*`*AuditGenerated*"]
]];
propertyFactoryExerciseRecords = symbolSummary /@ propertyFactoryExerciseNames;

safeInputForm[value_] := ToString[Unevaluated[value], InputForm];

algebraNames = If[
  MemberQ[afterPackageNames, "AlgMul`alg"],
  Quiet[Check[ToExpression["AlgMul`alg[\"algs\"]"], {}]],
  {}
];
algebraRecords = If[ListQ[algebraNames],
  Table[
    <|
      "name" -> ToString[name, InputForm],
      "characters" -> Quiet[Check[ToString[ToExpression["AlgMul`alg[\"chars\"][\"" <> ToString[name] <> "\"]"], InputForm], "UNAVAILABLE"]],
      "multiplication" -> Quiet[Check[ToString[ToExpression["AlgMul`alg[\"Mul\"][\"" <> ToString[name] <> "\"]"], InputForm], "UNAVAILABLE"]],
      "structureDot" -> Quiet[Check[ToString[Head[ToExpression["AlgMul`alg[\"StructDot\"][\"" <> ToString[name] <> "\"]"]], InputForm], "UNAVAILABLE"]]
    |>,
    {name, algebraNames}
  ],
  {}
];

evaluatedProductTable[symbolName_String, dimension_Integer] := Module[
  {basis = IdentityMatrix[dimension], function = ToExpression[symbolName]},
  Quiet[Check[
    Table[function[basis[[i]], basis[[j]]], {i, dimension}, {j, dimension}],
    "UNAVAILABLE"
  ]]
];

compositionTables = <|
  "C" -> evaluatedProductTable["AlgMul`CMul", 2],
  "Cs" -> evaluatedProductTable["AlgMul`CsMul", 2],
  "H" -> evaluatedProductTable["AlgMul`HMul", 4],
  "Hs" -> evaluatedProductTable["AlgMul`HsMul", 4],
  "O" -> evaluatedProductTable["AlgMul`OMul", 8],
  "Os" -> evaluatedProductTable["AlgMul`OsMul", 8]
|>;

loadStatus = Which[
  loadResult === $Aborted, "timeout",
  loadResult === $Failed, "failed",
  Global`contextAfterLoad =!= initialContext, "incomplete-context-not-restored",
  !MemberQ[afterPackageNames, "AlgMul`MakeProperty"], "completed-with-missing-required-symbols",
  True, "completed"
];

requiredSymbols = {
  "AlgMul`MakeChar", "AlgMul`MakeAlg", "AlgMul`MakeProperty",
  "AlgMul`OMul", "AlgMul`TMul", "AlgMul`MMul", "AlgMul`MTMul"
};
missingRequiredSymbols = Complement[requiredSymbols, afterPackageNames];

manifest = <|
  "schemaType" -> "AlgMulEvaluatedSurface",
  "schemaVersion" -> "0.1",
  "source" -> source,
  "sourceSHA256" -> ToUpperCase[IntegerString[FileHash[source, "SHA256"], 16, 64]],
  "kernelVersion" -> $Version,
  "kernelVersionNumber" -> ToString[$VersionNumber, InputForm],
  "started" -> started,
  "finished" -> finished,
  "loadStatus" -> loadStatus,
  "loadResultHead" -> ToString[Head[loadResult], InputForm],
  "initialContext" -> initialContext,
  "contextAfterLoad" -> Global`contextAfterLoad,
  "contextPathAfterLoad" -> Global`contextPathAfterLoad,
  "globalNamesBeforeCount" -> Length[beforeGlobalNames],
  "globalNamesAfterCount" -> Length[afterGlobalNames],
  "packageNamesBeforeCount" -> Length[beforePackageNames],
  "packageNamesAfterCount" -> Length[afterPackageNames],
  "newGlobalNameCount" -> Length[newGlobalNames],
  "newPackageNameCount" -> Length[newPackageNames],
  "definedNewNameCount" -> Length[definedNames],
  "newGlobalNames" -> newGlobalNames,
  "newPackageNames" -> newPackageNames,
  "requiredSymbols" -> requiredSymbols,
  "missingRequiredSymbols" -> missingRequiredSymbols,
  "symbolSummaries" -> summaries,
  "algebraRegistry" -> algebraRecords,
  "compositionTables" -> compositionTables,
  "expectedPropertySymbolCount" -> Length[expectedPropertySymbols],
  "propertySymbolsPresentCount" -> Count[Lookup[propertyRecords, "present"], True],
  "propertySymbolsDefinedCount" -> Count[Lookup[propertyRecords, "defined"], True],
  "propertySymbols" -> propertyRecords,
  "propertyFactoryExerciseResult" -> ToString[propertyFactoryExerciseResult, InputForm],
  "propertyFactoryExerciseNames" -> propertyFactoryExerciseNames,
  "propertyFactoryExerciseRecords" -> propertyFactoryExerciseRecords,
  "auditNotes" -> {
    "This is an evaluated surface inventory, not a correctness oracle.",
    "Quiet suppresses load messages in this bootstrap audit; a production audit must capture them explicitly.",
    "Generator exercise and representative behavior capture remain separate manifest phases."
  }
|>;

Export[output, manifest, "RawJSON", "Compact" -> False];
Print["ALG_MUL_AUDIT_OUTPUT=" <> output];
Print["LOAD_STATUS=" <> loadStatus];
Print["NEW_GLOBAL_NAMES=" <> ToString[Length[newGlobalNames]]];
Print["NEW_PACKAGE_NAMES=" <> ToString[Length[newPackageNames]]];
Print["DEFINED_NEW_NAMES=" <> ToString[Length[definedNames]]];
Print["REGISTERED_ALGEBRAS=" <> ToString[Length[algebraRecords]]];
Print["PROPERTY_SYMBOLS_DEFINED=" <> ToString[Count[Lookup[propertyRecords, "defined"], True]]];
