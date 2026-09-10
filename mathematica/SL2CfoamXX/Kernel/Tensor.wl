(* ::Package:: *)

(* SPDX-FileCopyrightText: 2026 Jeremy Martinon *)
(* SPDX-License-Identifier: GPL-3.0-or-later*)

(* Included in SL2CfoamXX`Private` after the shared LibraryLink helpers. *)

$tensorManager = "SL2CfoamXXTensor";
$tensorInitializations = {"zero", "uninitialized"};
$tensorLayouts = {"generic_column_major", "canonical_dense_column_major",
    "full_tensor_reassociated_by_i1"};
$tensorAccesses = {"read_only", "read_write"};
$tensorReassociatedLayouts = {"reflection_triangular_by_i1", "rectangular_by_i1"};
$tensorMintMaximum = 2^63 - 1;

createTensorObject[] := Module[{object},
    checked[initializeLibrary[]];
    object = Quiet[Check[
        CreateManagedLibraryExpression[$tensorManager, FullTensorObject], $Failed]];
    If[!TrueQ[ManagedLibraryExpressionQ[object, $tensorManager]],
        checked[failure["NativeTensorError", "Could not create a managed tensor owner."]]];
    object
];

tensorID[object_] := Module[{id},
    require[MatchQ[object, _FullTensorObject] &&
        TrueQ[ManagedLibraryExpressionQ[object, $tensorManager]],
        "Expected a live FullTensorObject."];
    id = ManagedLibraryExpressionID[object, $tensorManager];
    require[IntegerQ[id] && 0 < id <= $tensorMintMaximum,
        "The tensor's managed owner is no longer available."];
    id
];

(* This helper is only for failed construction/operation cleanup. Public
   objects have automatic lifetimes; callers cannot invalidate an alias by
   closing or clearing its native storage through this API. *)
discardTensorObject[object_] := AbortProtect[Module[{id},
    If[TrueQ[ManagedLibraryExpressionQ[object, $tensorManager]],
        id = ManagedLibraryExpressionID[object, $tensorManager];
        If[IntegerQ[id], Quiet[Check[callNative[38, {id}], Null]]]];
    Null
]];

(* WithCleanup protects owner creation and releases an incomplete result on
   Failure, Throw, Abort, or an external time/memory constraint. Operations
   fill an existing native destination exactly once and return metadata. *)
withNewTensor[operation_] := Module[{object = None, succeeded = False},
    WithCleanup[
        AbortProtect[object = createTensorObject[]],
        checked[operation[tensorID[object]]];
        succeeded = True;
        object,
        If[!succeeded, discardTensorObject[object]]
    ]
];

tensorShapeArgument[shape_] := Module[{dimensions, count},
    dimensions = integerVector[shape, 5, "tensor shape", $tensorMintMaximum];
    count = Times @@ dimensions;
    require[count <= Quotient[$tensorMintMaximum, 8],
        "The tensor payload is too large for a machine-sized array."];
    dimensions
];

tensorIndexArgument[indices_] :=
    integerVector[indices, 5, "tensor indices", $tensorMintMaximum];

tensorMetadataValue[data_] := Module[{scientific},
    If[!AssociationQ[data],
        checked[failure["ProtocolError", "Native tensor metadata is not an association."]]];
    scientific = data["scientific"];
    If[scientific =!= Null,
        If[!AssociationQ[scientific],
            checked[failure["ProtocolError", "Native scientific metadata is not an association."]]];
        scientific = <|
            "Spins" -> scientific["spins"],
            "Dl" -> scientific["Dl"],
            "Gamma" -> scientific["gamma"],
            "SimplicityConstraint" -> enumValue[scientific["simplicity_constraint"], $simplicity]
        |>,
        scientific = None
    ];
    <|
        "Shape" -> data["shape"],
        "Strides" -> data["strides"],
        "LogicalValues" -> data["logical_values"],
        "StoredValues" -> data["stored_values"],
        "StoredBytes" -> data["stored_bytes"],
        "DenseBytes" -> data["dense_bytes"],
        "Storage" -> enumValue[data["storage"], $storages],
        "Layout" -> enumValue[data["layout"], $tensorLayouts],
        "ReassociatedLayout" -> If[data["reassociated_layout"] === Null,
            None, enumValue[data["reassociated_layout"], $tensorReassociatedLayouts]],
        "Mapped" -> data["mapped"],
        "Writable" -> data["writable"],
        "SymmetryCompressed" -> data["symmetry_compressed"],
        "Dense" -> data["dense"],
        "Path" -> If[MemberQ[{Null, ""}, data["path"]], None, data["path"]],
        "Scientific" -> scientific
    |>
];

Options[CreateTensor] = {
    "Initialization" -> "zero", "Layout" -> "canonical_dense_column_major"};

CreateTensor[shape_, options : OptionsPattern[]] := api[Module[{dimensions, init, layout},
    validateOptions[CreateTensor, {options}];
    dimensions = tensorShapeArgument[shape];
    init = enumIndex[OptionValue["Initialization"], $tensorInitializations, "Initialization"];
    layout = enumIndex[Replace[OptionValue["Layout"],
        "full_tensor_reassociated_by_a1" -> "full_tensor_reassociated_by_i1"],
        $tensorLayouts, "Layout"];
    withNewTensor[Function[id, callNative[20, Join[{id, init, layout}, dimensions]]]]
]];

CreateTensor[arguments___] := failure["InvalidArgument",
    "CreateTensor expects a list of five nonnegative integer dimensions and optional rules."];

Options[CreateMappedTensor] = Options[CreateTensor];

CreateMappedTensor[path_, shape_, options : OptionsPattern[]] := api[Module[
    {file, dimensions, init, layout},
    validateOptions[CreateMappedTensor, {options}];
    file = pathString[path, "path"];
    require[file =!= "", "A mapped tensor requires an explicit file path."];
    dimensions = tensorShapeArgument[shape];
    init = enumIndex[OptionValue["Initialization"], $tensorInitializations, "Initialization"];
    layout = enumIndex[Replace[OptionValue["Layout"],
        "full_tensor_reassociated_by_a1" -> "full_tensor_reassociated_by_i1"],
        $tensorLayouts, "Layout"];
    withNewTensor[Function[id, callNative[21, Join[{id, init, layout}, dimensions], {}, file]]]
]];

CreateMappedTensor[arguments___] := failure["InvalidArgument",
    "CreateMappedTensor expects a path, five dimensions, and optional rules."];

TensorFromArray[array_] := api[Module[{dimensions, values},
    require[ListQ[array] && ArrayDepth[array] == 5,
        "TensorFromArray expects a rectangular rank-five array."];
    dimensions = tensorShapeArgument[Dimensions[array]];
    values = Flatten[array];
    require[Length[values] == Times @@ dimensions,
        "TensorFromArray expects a rectangular rank-five array."];
    values = finiteReal[#, "tensor value"] & /@ values;
    withNewTensor[Function[id, callNative[40, Join[{id}, dimensions], values]]]
]];

TensorFromArray[arguments___] := failure["InvalidArgument",
    "TensorFromArray expects one rectangular rank-five array."];

LoadTensor[path_] := api[Module[{file = pathString[path, "path"]},
    require[file =!= "", "Loading a tensor requires an explicit file path."];
    withNewTensor[Function[id, callNative[22, {id}, {}, file]]]
]];

LoadTensor[arguments___] := failure["InvalidArgument", "LoadTensor expects one file path."];

Options[MapTensor] = {"Access" -> "read_only"};

MapTensor[path_, options : OptionsPattern[]] := api[Module[{file, access},
    validateOptions[MapTensor, {options}];
    file = pathString[path, "path"];
    require[file =!= "", "Mapping a tensor requires an explicit file path."];
    access = enumIndex[OptionValue["Access"], $tensorAccesses, "Access"];
    withNewTensor[Function[id, callNative[23, {id, access}, {}, file]]]
]];

MapTensor[arguments___] := failure["InvalidArgument",
    "MapTensor expects a file path and optional Access rule."];

CloneTensor[object_] := api[Module[{source = tensorID[object]},
    withNewTensor[Function[id, callNative[31, {id, source}]]]
]];

CloneTensor[arguments___] := failure["InvalidArgument", "CloneTensor expects one tensor."];

DenseCopyTensor[object_] := api[Module[{source = tensorID[object]},
    withNewTensor[Function[id, callNative[32, {id, source}]]]
]];

DenseCopyTensor[arguments___] := failure["InvalidArgument", "DenseCopyTensor expects one tensor."];

SaveTensor[object_, path_] := api[Module[{id, file},
    id = tensorID[object];
    file = pathString[path, "path"];
    require[file =!= "", "Saving a tensor requires an explicit file path."];
    checked[callNative[33, {id}, {}, file]]
]];

SaveTensor[arguments___] := failure["InvalidArgument", "SaveTensor expects a tensor and file path."];

FlushTensor[object_] := api[checked[callNative[34, {tensorID[object]}]]];

FlushTensor[arguments___] := failure["InvalidArgument", "FlushTensor expects one tensor."];

Options[ConvertTensorStorage] = {"OverwriteExisting" -> False};

ConvertTensorStorage[object_, storage_, path_, options : OptionsPattern[]] := api[Module[
    {source, destination, file, overwrite},
    validateOptions[ConvertTensorStorage, {options}];
    source = tensorID[object];
    destination = enumIndex[storage, $storages, "storage"];
    file = pathString[path, "path"];
    require[file =!= "", "Storage conversion requires an explicit destination file path."];
    overwrite = OptionValue["OverwriteExisting"];
    require[BooleanQ[overwrite], "OverwriteExisting must be True or False."];
    withNewTensor[Function[id,
        callNative[35, {id, source, destination, Boole[overwrite]}, {}, file]]]
]];

ConvertTensorStorage[arguments___] := failure["InvalidArgument",
    "ConvertTensorStorage expects a tensor, storage mode, destination path, and optional rules."];

TensorMetadata[path_String] := api[Module[{file = pathString[path, "path"]},
    require[file =!= "", "Tensor metadata requires an explicit file path."];
    tensorMetadataValue[checked[callNative[25, {}, {}, file]]]
]];

TensorMetadata[object_] := api[
    tensorMetadataValue[checked[callNative[24, {tensorID[object]}]]]
];

TensorMetadata[arguments___] := failure["InvalidArgument", "TensorMetadata expects a tensor or file path."];

TensorShape[object_] := api[checked[TensorMetadata[object]]["Shape"]];

TensorShape[arguments___] := failure["InvalidArgument", "TensorShape expects a tensor or file path."];

TensorValue[object_, indices_] := api[Module[{id, positions},
    id = tensorID[object];
    positions = tensorIndexArgument[indices];
    (* The native wire also uses one-based canonical coordinates. *)
    checked[callNative[26, Join[{id}, positions]]]
]];

TensorValue[arguments___] := failure["InvalidArgument",
    "TensorValue expects a tensor and a list of five one-based integer indices."];

SetTensorValue[object_, indices_, value_] := api[Module[{id, positions, scalar},
    id = tensorID[object];
    positions = tensorIndexArgument[indices];
    scalar = finiteReal[value, "tensor value"];
    checked[callNative[27, Join[{id}, positions], {scalar}]];
    object
]];

SetTensorValue[arguments___] := failure["InvalidArgument",
    "SetTensorValue expects a tensor, five one-based integer indices, and a real value."];

tensorArrayValue[object_, physical_] := Module[{id, result},
    id = tensorID[object];
    (* Dispatch supplies the ordinary native validation/error classification.
       The typed array function then allocates an independent Wolfram tensor. *)
    checked[callNative[24, {id}]];
    result = Quiet[Check[nativeTensorArray[id, physical], $Failed]];
    If[!ListQ[result],
        checked[failure["NativeArrayError",
            "Could not copy the native tensor into a Wolfram array.", <|"Cause" -> result|>]]];
    result
];

TensorArray[object_] := api[tensorArrayValue[object, False]];
StoredTensorValues[object_] := api[tensorArrayValue[object, True]];

TensorArray[arguments___] := failure["InvalidArgument", "TensorArray expects one tensor."];
StoredTensorValues[arguments___] := failure["InvalidArgument", "StoredTensorValues expects one tensor."];

(* FullTensorObject is a managed resource, not a List. Normal is the explicit
   bridge to native Wolfram array operations; no Part or Set[Part] overrides
   reinterpret the managed ID as a tensor coordinate. *)
FullTensorObject /: Normal[object_FullTensorObject] := TensorArray[object];
FullTensorObject /: Dimensions[object_FullTensorObject] := TensorShape[object];

TensorTag[object_] := api[ByteArray[checked[callNative[37, {tensorID[object]}]]]];

TensorTag[arguments___] := failure["InvalidArgument", "TensorTag expects one tensor."];

Options[FormatTensor] = {
    "MaxValues" -> 4096, "Precision" -> 17, "IncludeMetadata" -> True};

FormatTensor[object_, options : OptionsPattern[]] := api[Module[
    {id, maximum, precision, includeMetadata},
    validateOptions[FormatTensor, {options}];
    id = tensorID[object];
    maximum = OptionValue["MaxValues"];
    maximum = If[maximum === All || maximum === Infinity, -1,
        nonnegativeInteger[maximum, "MaxValues", $tensorMintMaximum]];
    precision = nonnegativeInteger[OptionValue["Precision"], "Precision", 2^31 - 1];
    includeMetadata = OptionValue["IncludeMetadata"];
    require[BooleanQ[includeMetadata], "IncludeMetadata must be True or False."];
    checked[callNative[36, {id, maximum, precision, Boole[includeMetadata]}]]
]];

FormatTensor[arguments___] := failure["InvalidArgument",
    "FormatTensor expects a tensor and optional formatting rules."];
