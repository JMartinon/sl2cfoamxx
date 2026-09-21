(* ::Package:: *)

(* SPDX-FileCopyrightText: 2026 Jeremy Martinon *)
(* SPDX-License-Identifier: GPL-3.0-or-later*)

BeginPackage["SL2CfoamXX`"];

DefaultConfig::usage = "DefaultConfig[] returns the native build-dependent default configuration without creating the cache directory.";
CurrentConfig::usage = "CurrentConfig[] returns an independent Association snapshot of the active native configuration.";
Configure::usage = "Configure[config] applies a complete configuration Association returned by DefaultConfig or CurrentConfig. Active computations retain their snapshot.";
LibraryVersion::usage = "LibraryVersion[] returns the version of the loaded native library.";
BuildInfo::usage = "BuildInfo[] returns compiler, BLAS, CUDA, and other native build information as an Association.";
BackendAvailableQ::usage = "BackendAvailableQ[\"CPU\"|\"GPU\"] tests native backend availability without selecting a fallback.";
CUDADeviceCount::usage = "CUDADeviceCount[] returns the number of visible CUDA devices; zero for CPU-only builds.";
CUDADeviceInfo::usage = "CUDADeviceInfo[ordinal] returns native device information. CUDA ordinals are zero-based.";
$LorentzianBoundaryAngles::usage = "$LorentzianBoundaryAngles gives the library's Lorentzian preset as a packed 20 by 2 real array of {theta,phi} pairs in radians, in canonical oriented-normal order. It closes for doubled spins {10,10,10,10,4,4,4,4,4,4} and common positive scalings. Assign it to a variable before editing.";
$EuclideanBoundaryAngles::usage = "$EuclideanBoundaryAngles gives the library's Euclidean preset as a packed 20 by 2 real array of {theta,phi} pairs in radians, in canonical oriented-normal order. It closes for equal boundary spins. Assign it to a variable before editing.";
ComputeAmplitude::usage = "ComputeAmplitude[spins, intertwiners, Dl, gamma, \"CachePolicy\" -> \"disabled\"] computes one real fixed-intertwiner amplitude on CPU. Spin labels are doubled integers.";
ComputeCoherentAmplitude::usage = "ComputeCoherentAmplitude[spins, Dl, gamma, angles, options] computes a complex CPU/GPU coherent amplitude from 20 {theta,phi} pairs. ComputeCoherentAmplitude[spins, Dl, gamma, C1, C2, C3, C4, C5, options] contracts caller-supplied finite complex vectors in canonical intertwiner order, without conjugation or normalization. Options are \"Backend\" -> Automatic and \"ProgressFunction\" -> None.";
ComputeFullTensor::usage = "ComputeFullTensor[spins, Dl, gamma, \"Backend\" -> Automatic, \"ProgressFunction\" -> None, \"ReturnType\" -> \"Tensor\"] returns a managed FullTensorObject; ReturnType -> \"Array\" returns an independent canonical packed array.";
ComputeCoherentCoefficients::usage = "ComputeCoherentCoefficients[spins4, angles4] returns the four-valent coherent coefficient vector as packed machine complex values.";
CheckClosure::usage = "CheckClosure[spins, angles, \"AbsoluteTolerance\" -> 10^-12, \"RelativeTolerance\" -> 10^-12] reports closure for all five nodes without printing.";
IndicesFromIntertwiners::usage = "IndicesFromIntertwiners[spins, intertwiners] returns five one-based canonical tensor indices from doubled integer labels.";
FullTensorObject::usage = "FullTensorObject is a managed native tensor. Normal[tensor] makes an independent packed array; use TensorValue for canonical one-based indexing.";
CreateTensor::usage = "CreateTensor[shape, \"Initialization\" -> \"zero\", \"Layout\" -> \"canonical_dense_column_major\"] creates an owned rank-five tensor.";
CreateMappedTensor::usage = "CreateMappedTensor[path, shape, options] creates a writable mapped rank-five tensor.";
TensorFromArray::usage = "TensorFromArray[array] copies a rank-five real Wolfram array into an independent native canonical tensor.";
CloneTensor::usage = "CloneTensor[tensor] makes an independent writable native copy in the same representation.";
DenseCopyTensor::usage = "DenseCopyTensor[tensor] makes an independent native canonical dense copy.";
SaveTensor::usage = "SaveTensor[tensor, path] saves a tensor in the native format shared with C++, Python, and Julia.";
LoadTensor::usage = "LoadTensor[path] loads an independent writable native tensor.";
MapTensor::usage = "MapTensor[path, \"Access\" -> \"read_only\"] maps a native tensor file. Read-only mappings reject mutation.";
FlushTensor::usage = "FlushTensor[tensor] flushes a writable mapped tensor to its file.";
ConvertTensorStorage::usage = "ConvertTensorStorage[tensor, storage, path, \"OverwriteExisting\" -> False] makes a lossless storage conversion while preserving scientific metadata.";
TensorMetadata::usage = "TensorMetadata[tensor|path] returns logical shape, physical storage, mapping, and optional scientific metadata as an Association.";
TensorShape::usage = "TensorShape[tensor] returns the five logical dimensions. Dimensions[tensor] is equivalent.";
TensorValue::usage = "TensorValue[tensor, {i1,i2,i3,i4,i5}] reads one canonical value using one-based indices.";
SetTensorValue::usage = "SetTensorValue[tensor, indices, value] writes one canonical value using one-based indices.";
TensorArray::usage = "TensorArray[tensor] returns an independent, canonical rank-five packed real array. Normal[tensor] is equivalent.";
StoredTensorValues::usage = "StoredTensorValues[tensor] returns an independent packed vector in native physical storage order, which can differ from canonical tensor order.";
TensorTag::usage = "TensorTag[tensor] returns an independent ByteArray containing the 1024-byte native tag.";
FormatTensor::usage = "FormatTensor[tensor, \"MaxValues\" -> 4096, \"Precision\" -> 17, \"IncludeMetadata\" -> True] returns bounded native diagnostic text. Diagnostic coordinate labels are zero-based.";

Begin["`Private`"];

$kernelDirectory = DirectoryName[$InputFileName];
$packageDirectory = DirectoryName[$kernelDirectory];
$libraryReady = False;
$progressBarEnabled = False;
$failureTag = Unique["SL2CfoamXXFailure"];
$backends = {"CPU", "GPU"};
$storages = {"in_memory_sym", "in_memory_dense", "on_disk_sym", "on_disk_dense"};
$simplicity = {"gamma_j", "gamma_j_p1"};
$verbosity = {"silent", "summary", "timing", "detailed"};
$printingStyles = {"UTF8", "ASCII"};
$finalTiles = {"automatic", "tile_4x4", "tile_4x8", "tile_8x4"};

failure[tag_String, message_String, data_Association : <||>] :=
  Failure[tag, Join[<|"MessageTemplate" -> message, "Code" -> tag|>, data]];
SetAttributes[api, HoldAll];
api[body_] := Catch[body, $failureTag];
checked[value_] := If[FailureQ[value], Throw[value, $failureTag], value];
require[condition_, message_String, data_Association : <||>] :=
  If[!TrueQ[condition], Throw[failure["InvalidArgument", message, data], $failureTag]];

nonnegativeInteger[value_, name_String, maximum_Integer : 2^63 - 1] := (
  require[IntegerQ[value] && 0 <= value <= maximum,
    name <> " must be a nonnegative exact integer no larger than " <> ToString[maximum] <> "."];
  value
);
integerVector[values_, count_Integer, name_String, maximum_Integer : 2^31 - 1] := (
  require[ListQ[values] && Length[values] == count,
    name <> " must be a list of " <> ToString[count] <> " exact integer labels."];
  nonnegativeInteger[#, name, maximum] & /@ values
);
finiteReal[value_, name_String] := Module[{result},
  result = Quiet[Check[N[value, MachinePrecision], $Failed]];
  require[Head[result] === Real && MachineNumberQ[result], name <> " must be a finite real number."];
  result
];
pathString[value_, name_String] := Module[{result = value},
  If[result === None || result === Automatic || result === Null, Return[""]];
  require[StringQ[result] && StringFreeQ[result, FromCharacterCode[0]],
    name <> " must be a path string without NUL, None, or Automatic."];
  result
];
enumIndex[value_, choices_List, name_String] := Module[{position},
  require[StringQ[value], name <> " must be a string."];
  position = FirstPosition[choices, value, Missing["NotFound"]];
  require[!MissingQ[position], name <> " must be one of " <> ToString[choices, InputForm] <> "."];
  First[position] - 1
];
enumValue[index_Integer, choices_List] := (
  require[0 <= index < Length[choices], "The native library returned an unknown enum value."];
  choices[[index + 1]]
);
booleanInteger[value_, name_String] := (
  require[value === True || value === False, name <> " must be True or False."];
  Boole[value]
);
validateOptions[symbol_, rules_List] := Module[{flat = Flatten[rules], unknown},
  require[AllTrue[flat, MatchQ[#, _Rule | _RuleDelayed] &], "Options must be rules."];
  unknown = Complement[First /@ flat, First /@ Options[symbol]];
  require[unknown === {}, "Unknown option names.", <|"Options" -> unknown|>]
];

(* LibraryLink decodes UTF8String into a Wolfram Unicode string. RawJSON must
   receive UTF-8 bytes explicitly: its ImportString path rejects characters
   such as Greek letters instead of encoding them as JSON input bytes. *)
parseNativeJSON[raw_String] :=
  ImportByteArray[ByteArray[ToCharacterCode[raw, "UTF8"]], "RawJSON"];

initializeLibrary[] := Module[{override, extension, path, dispatch, tensorArray, protocol},
  If[TrueQ[$libraryReady], Return[Null]];
  override = Quiet[Check[Environment["SL2CFOAMXX_MATHEMATICA_LIBRARY"], $Failed]];
  extension = Switch[$OperatingSystem, "MacOSX", ".dylib", "Windows", ".dll", _, ".so"];
  path = If[StringQ[override] && StringLength[override] > 0, ExpandFileName[override],
    FileNameJoin[{$packageDirectory, "LibraryResources", $SystemID,
      "libsl2cfoamxx_mathematica" <> extension}]];
  If[!FileExistsQ[path], Return[failure["LibraryNotFound",
    "The compiled Mathematica bridge was not found. Load a built or installed SL2CfoamXX package.",
    <|"Path" -> path|>]]];
  dispatch = Quiet[Check[LibraryFunctionLoad[path, "sfx_dispatch",
    {Integer, {Integer, 1, "Constant"}, {Real, 1, "Constant"}, "UTF8String", "UTF8String"},
    "UTF8String"], $Failed]];
  tensorArray = Quiet[Check[LibraryFunctionLoad[path, "sfx_tensor_array",
    {Integer, "Boolean"}, {Real, _}], $Failed]];
  If[Head[dispatch] =!= LibraryFunction || Head[tensorArray] =!= LibraryFunction,
    Return[failure["LibraryLoadError", "The Mathematica bridge or a native dependency could not be loaded.",
      <|"Path" -> path|>]]];
  nativeDispatch = dispatch;
  nativeTensorArray = tensorArray;
  protocol = Quiet[Check[parseNativeJSON[nativeDispatch[0, {}, {}, "", ""]], $Failed]];
  If[!AssociationQ[protocol] || !TrueQ[Lookup[protocol, "ok", False]] ||
      Lookup[protocol, "value", Missing[]] =!= 1,
    Return[failure["ProtocolMismatch", "The loaded bridge has an incompatible protocol.", <|"Path" -> path|>]]];
  $libraryReady = True;
  Null
];

$nativeErrorNames = {"backend_unavailable", "backend_failure", "invalid_tensor_file",
  "cache_failure", "resource_limit", "numerical_failure", "internal_failure", "unsupported_operation"};
$nativeErrorTags = {"BackendUnavailable", "BackendFailure", "InvalidTensorFile",
  "CacheFailure", "ResourceLimit", "NumericalFailure", "InternalFailure", "UnsupportedOperation"};
nativeFailure[record_Association] := Module[{kind, code, tag, codeName},
  kind = Lookup[record, "kind", "Runtime"];
  code = Lookup[record, "code", 0];
  tag = If[kind === "Library" && IntegerQ[code] && 1 <= code <= Length[$nativeErrorTags],
    $nativeErrorTags[[code]], kind];
  codeName = If[kind === "Library" && IntegerQ[code] && 1 <= code <= Length[$nativeErrorNames],
    $nativeErrorNames[[code]], tag];
  failure[tag, Lookup[record, "message", "Native library operation failed."],
    <|"Code" -> codeName, "NativeCode" -> code, "Kind" -> kind,
      "SystemCode" -> Lookup[record, "system_code", 0],
      "Path" -> Replace[Lookup[record, "path1", Null], {Null | "" -> None}],
      "Path2" -> Replace[Lookup[record, "path2", Null], {Null | "" -> None}]|>]
];
callNative[opcode_Integer, integers_List : {}, reals_List : {}, text1_String : "", text2_String : ""] :=
 Module[{loaded, raw, envelope},
  loaded = initializeLibrary[];
  If[FailureQ[loaded], Return[loaded]];
  raw = Quiet[Check[nativeDispatch[opcode, integers, reals, text1, text2], $Failed]];
  If[!StringQ[raw], Return[failure["LibraryCallError", "The Mathematica bridge could not marshal this call."]]];
  envelope = Quiet[Check[parseNativeJSON[raw], $Failed]];
  If[!AssociationQ[envelope], Return[failure["ProtocolError", "The bridge returned an invalid response."]]];
  Which[
    TrueQ[Lookup[envelope, "ok", False]], Lookup[envelope, "value", Null],
    AssociationQ[Lookup[envelope, "error", None]], nativeFailure[envelope["error"]],
    True, failure["ProtocolError", "The bridge response has no result or error."]]
 ];

configFromNative[record_Association] := Module[{v = record["ints"], p = record["reals"]},
  require[Length[v] == 17 && Length[p] == 1, "The native configuration record has an invalid length."];
  <|"Backend" -> enumValue[v[[1]], $backends], "ProgressBar" -> (v[[2]] != 0),
    "SimplicityConstraint" -> enumValue[v[[3]], $simplicity],
    "CPUThreads" -> v[[14]], "MemoryBudgetBytes" -> v[[15]],
    "Verbosity" -> enumValue[v[[4]], $verbosity], "PrintingStyle" -> enumValue[v[[5]], $printingStyles],
    "Tensors" -> <|"Directory" -> Replace[record["directory"], "" -> Automatic],
      "Storage" -> enumValue[v[[6]], $storages],
      "OutputPath" -> Replace[record["output_path"], {"" | Null -> None}],
      "ReuseExisting" -> (v[[7]] != 0), "MaximumPayloadSizeBytes" -> v[[16]]|>,
    "CoherentAmplitude" -> <|"CheckClosure" -> (v[[8]] != 0), "PruningRatio" -> N[First[p], MachinePrecision]|>,
    "LowLevel" -> <|"BoosterRhoTile" -> v[[10]], "FullTensorGPURowCapacity" -> v[[17]],
      "CoherentGPUPipelineSlots" -> v[[11]], "FullTensorGPUProducerWavesPerSM" -> v[[12]],
      "FullTensorCPUPBatch" -> v[[13]], "FullTensorCPUFinalTile" -> enumValue[v[[9]], $finalTiles]|>|>
];
DefaultConfig[] := api[configFromNative[checked[callNative[1]]]];
CurrentConfig[] := api[Module[{config = configFromNative[checked[callNative[2]]]},
  $progressBarEnabled = config["ProgressBar"]; config]];

exactKeys[association_, keys_List, name_String] := (
  require[AssociationQ[association], name <> " must be an Association."];
  require[Sort[Keys[association]] === Sort[keys], name <> " has missing or unknown fields.",
    <|"ExpectedKeys" -> keys, "ActualKeys" -> Keys[association]|>]
);
Configure[config_] := api[Module[{tensors, coherent, low, integers, pruning, directory, output, result},
  exactKeys[config, {"Backend", "ProgressBar", "SimplicityConstraint", "CPUThreads", "MemoryBudgetBytes",
    "Verbosity", "PrintingStyle", "Tensors", "CoherentAmplitude", "LowLevel"}, "configuration"];
  tensors = config["Tensors"]; coherent = config["CoherentAmplitude"]; low = config["LowLevel"];
  exactKeys[tensors, {"Directory", "Storage", "OutputPath", "ReuseExisting", "MaximumPayloadSizeBytes"}, "Tensors"];
  exactKeys[coherent, {"CheckClosure", "PruningRatio"}, "CoherentAmplitude"];
  exactKeys[low, {"BoosterRhoTile", "FullTensorGPURowCapacity", "CoherentGPUPipelineSlots",
    "FullTensorGPUProducerWavesPerSM", "FullTensorCPUPBatch", "FullTensorCPUFinalTile"}, "LowLevel"];
  integers = {
    enumIndex[config["Backend"], $backends, "Backend"], booleanInteger[config["ProgressBar"], "ProgressBar"],
    enumIndex[config["SimplicityConstraint"], $simplicity, "SimplicityConstraint"],
    enumIndex[config["Verbosity"], $verbosity, "Verbosity"],
    enumIndex[config["PrintingStyle"], $printingStyles, "PrintingStyle"],
    enumIndex[tensors["Storage"], $storages, "Storage"], booleanInteger[tensors["ReuseExisting"], "ReuseExisting"],
    booleanInteger[coherent["CheckClosure"], "CheckClosure"], enumIndex[low["FullTensorCPUFinalTile"], $finalTiles, "FullTensorCPUFinalTile"],
    nonnegativeInteger[low["BoosterRhoTile"], "BoosterRhoTile", 2^32 - 1],
    nonnegativeInteger[low["CoherentGPUPipelineSlots"], "CoherentGPUPipelineSlots", 2^32 - 1],
    nonnegativeInteger[low["FullTensorGPUProducerWavesPerSM"], "FullTensorGPUProducerWavesPerSM", 2^32 - 1],
    nonnegativeInteger[low["FullTensorCPUPBatch"], "FullTensorCPUPBatch", 2^32 - 1],
    nonnegativeInteger[config["CPUThreads"], "CPUThreads"], nonnegativeInteger[config["MemoryBudgetBytes"], "MemoryBudgetBytes"],
    nonnegativeInteger[tensors["MaximumPayloadSizeBytes"], "MaximumPayloadSizeBytes"],
    nonnegativeInteger[low["FullTensorGPURowCapacity"], "FullTensorGPURowCapacity"]};
  pruning = finiteReal[coherent["PruningRatio"], "PruningRatio"];
  require[0. <= pruning <= 1., "PruningRatio must be between zero and one."];
  directory = pathString[tensors["Directory"], "Directory"];
  output = pathString[tensors["OutputPath"], "OutputPath"];
  result = checked[callNative[3, integers, {pruning}, directory, output]];
  $progressBarEnabled = config["ProgressBar"];
  result
]];

renameFields[record_Association, names_Association] := Association @
  KeyValueMap[Lookup[names, #1, #1] -> #2 &, record];
$buildNames = <|"version" -> "Version", "compiler_id" -> "CompilerID", "compiler_version" -> "CompilerVersion",
  "build_type" -> "BuildType", "blas_vendor_selector" -> "BLASVendorSelector", "blas_provider" -> "BLASProvider",
  "native_arch_enabled" -> "NativeArchEnabled", "cuda_backend_built" -> "CUDABackendBuilt",
  "cuda_fast_math_requested" -> "CUDAFastMathRequested", "cuda_toolkit_version" -> "CUDAToolkitVersion",
  "cuda_architectures" -> "CUDAArchitectures", "cutensor_version" -> "CuTENSORVersion"|>;
$deviceNames = <|"ordinal" -> "Ordinal", "name" -> "Name", "compute_capability_major" -> "ComputeCapabilityMajor",
  "compute_capability_minor" -> "ComputeCapabilityMinor", "total_memory_bytes" -> "TotalMemoryBytes",
  "multiprocessor_count" -> "MultiprocessorCount", "warp_size" -> "WarpSize", "max_threads_per_block" -> "MaxThreadsPerBlock",
  "pci_bus_id" -> "PCIBusID"|>;
BuildInfo[] := api[renameFields[checked[callNative[4]], $buildNames]];
LibraryVersion[] := api[checked[BuildInfo[]]["Version"]];
BackendAvailableQ[backend_] := api[callNative[5, {enumIndex[backend, $backends, "Backend"]}]];
CUDADeviceCount[] := callNative[6];
CUDADeviceInfo[ordinal_ : 0] := api[renameFields[checked[callNative[7,
  {nonnegativeInteger[ordinal, "CUDA device ordinal"]}]], $deviceNames]];

$LorentzianBoundaryAngles := api[Developer`ToPackedArray[checked[callNative[8, {0}]], Real]];
$EuclideanBoundaryAngles := api[Developer`ToPackedArray[checked[callNative[8, {1}]], Real]];

anglesVector[angles_, rows_Integer] := (
  require[MatrixQ[angles] && Dimensions[angles] === {rows, 2},
    "Angles must be a " <> ToString[rows] <> " by 2 list of {theta,phi} pairs."];
  finiteReal[#, "angle"] & /@ Flatten[angles]
);
complexValue[pair_List] := N[First[pair], MachinePrecision] + I N[Last[pair], MachinePrecision];
complexVector[values_, name_String] := (
  require[ListQ[values] && AllTrue[values, NumericQ],
    name <> " must be a vector of finite real or complex numbers."];
  {finiteReal[Re[#], name <> " real part"], finiteReal[Im[#], name <> " imaginary part"]} & /@ values
);

Options[ComputeAmplitude] = {"CachePolicy" -> "disabled"};
ComputeAmplitude[spins_, intertwiners_, Dl_, gamma_, opts : OptionsPattern[]] := api[
 Module[{s, a, cutoff, parameter, policy},
  validateOptions[ComputeAmplitude, {opts}];
  s = integerVector[spins, 10, "spins"]; a = integerVector[intertwiners, 5, "intertwiners"];
  cutoff = nonnegativeInteger[Dl, "Dl"]; parameter = finiteReal[gamma, "gamma"];
  policy = enumIndex[OptionValue["CachePolicy"], {"disabled", "prefer_full_tensor"}, "CachePolicy"];
  N[checked[callNative[10, Join[{cutoff, policy}, s, a], {parameter}]], MachinePrecision]
 ]];
IndicesFromIntertwiners[spins_, intertwiners_] := api[callNative[15,
  Join[integerVector[spins, 10, "spins"], integerVector[intertwiners, 5, "intertwiners"]]]];
ComputeCoherentCoefficients[spins_, angles_] := api[Module[{s, normals, result},
  s = integerVector[spins, 4, "tetrahedron spins"]; normals = anglesVector[angles, 4];
  result = checked[callNative[13, s, normals]];
  Developer`ToPackedArray[complexValue /@ result, Complex]
]];
Options[CheckClosure] = {"AbsoluteTolerance" -> 10.^-12, "RelativeTolerance" -> 10.^-12};
CheckClosure[spins_, angles_, opts : OptionsPattern[]] := api[Module[{s, normals, absolute, relative, result},
  validateOptions[CheckClosure, {opts}];
  s = integerVector[spins, 10, "spins"]; normals = anglesVector[angles, 20];
  absolute = finiteReal[OptionValue["AbsoluteTolerance"], "AbsoluteTolerance"];
  relative = finiteReal[OptionValue["RelativeTolerance"], "RelativeTolerance"];
  result = checked[callNative[14, s, Join[{absolute, relative}, normals]]];
  <|"AllClosed" -> result["all_closed"], "Nodes" ->
    (renameFields[#, <|"vector" -> "Vector", "residual_norm" -> "ResidualNorm", "scale" -> "Scale",
      "relative_residual" -> "RelativeResidual", "closed" -> "Closed"|>] & /@ result["nodes"])|>
]];

Get[FileNameJoin[{$kernelDirectory, "Tensor.wl"}]];

notebookQ[] := TrueQ[$Notebooks] && MatchQ[$FrontEnd, _FrontEndObject];
progressSettings[backend_, callback_] := Module[{selected, display},
  selected = If[backend === Automatic, -1, enumIndex[backend, $backends, "Backend"]];
  If[callback =!= None,
    require[selected != -1, "ProgressFunction requires an explicit Backend."];
    require[MatchQ[callback, _Function] || (Head[callback] === Symbol &&
      !MemberQ[{True, False, Null, Automatic, $Failed}, callback]),
      "ProgressFunction must be a function or None."]];
  display = TrueQ[$progressBarEnabled] && notebookQ[];
  If[display && selected == -1, selected = enumIndex[checked[CurrentConfig[]]["Backend"], $backends, "Backend"]];
  {selected, callback =!= None || display, display}
];
progressEvent[event_Association] := <|
  "Operation" -> enumValue[event["operation"], {"coherent_amplitude", "full_tensor"}],
  "Phase" -> enumValue[event["phase"], {"contraction", "streamed_computation"}],
  "Backend" -> enumValue[event["backend"], $backends],
  "Status" -> enumValue[event["status"], {"started", "running", "restarted", "completed", "stopped"}],
  "Attempt" -> event["attempt"], "Completed" -> event["completed"], "Total" -> event["total"],
  "ElapsedSeconds" -> event["elapsed_seconds"], "ETASeconds" -> event["eta_seconds"],
  "ETAAvailable" -> event["eta_available"]|>;
invokeProgress[callback_, event_Association] := Module[{value = callback[event]},
  If[value === $Failed,
    checked[failure["ProgressFunctionError", "The progress function returned $Failed."]], checked[value]]
];

(* Only this optional path creates a worker. The job owns copied inputs and
   native output. Cleanup joins it before user Throw or Abort can escape. *)
runJob[startOpcode_Integer, integers_List, reals_List, callback_, display_] :=
 Module[{job = None, id = None, batch, latest = None, event, done = False, result},
  checked[initializeLibrary[]];
  WithCleanup[
    AbortProtect[
      job = CreateManagedLibraryExpression["SL2CfoamXXJob", nativeJob];
      require[ManagedLibraryExpressionQ[job, "SL2CfoamXXJob"], "The native progress job could not be allocated."];
      id = ManagedLibraryExpressionID[job, "SL2CfoamXXJob"]
    ],
    (
      checked[callNative[startOpcode, Prepend[integers, id], reals]];
      result = If[TrueQ[display],
        Monitor[
          While[!done,
            batch = checked[callNative[52, {id}]];
            Do[event = progressEvent[raw]; latest = event;
              If[callback =!= None, invokeProgress[callback, event]], {raw, batch["events"]}];
            done = TrueQ[batch["done"]]; If[!done, Pause[.01]]
          ]; checked[callNative[53, {id}]],
          If[AssociationQ[latest],
            Column[{latest["Operation"] <> " (" <> latest["Backend"] <> ")",
              ProgressIndicator[latest["Completed"], {0, Max[1, latest["Total"]]}],
              Row[{latest["Completed"], " / ", latest["Total"], "   elapsed ",
                NumberForm[latest["ElapsedSeconds"], {8, 1}], " s"}]}], "Preparing native computation..."]
        ],
        While[!done,
          batch = checked[callNative[52, {id}]];
          Do[event = progressEvent[raw];
            If[callback =!= None, invokeProgress[callback, event]], {raw, batch["events"]}];
          done = TrueQ[batch["done"]]; If[!done, Pause[.01]]
        ]; checked[callNative[53, {id}]]
      ]; result
    ),
    AbortProtect[If[IntegerQ[id], Quiet[callNative[54, {id}]]]]
  ]
 ];

Options[ComputeCoherentAmplitude] = {"Backend" -> Automatic, "ProgressFunction" -> None};
ComputeCoherentAmplitude[spins_, Dl_, gamma_, angles_, opts : OptionsPattern[]] := api[
 Module[{s, cutoff, parameter, normals, callback, settings, integers, reals, result},
  validateOptions[ComputeCoherentAmplitude, {opts}];
  s = integerVector[spins, 10, "spins"]; cutoff = nonnegativeInteger[Dl, "Dl"];
  parameter = finiteReal[gamma, "gamma"]; normals = anglesVector[angles, 20];
  callback = OptionValue["ProgressFunction"]; settings = progressSettings[OptionValue["Backend"], callback];
  integers = Join[{cutoff, First[settings]}, s]; reals = Prepend[normals, parameter];
  result = checked[If[TrueQ[settings[[2]]], runJob[51, integers, reals, callback, settings[[3]]],
    callNative[11, integers, reals]]];
  complexValue[result]
 ]];
ComputeCoherentAmplitude[spins_, Dl_, gamma_, c1_, c2_, c3_, c4_, c5_, opts : OptionsPattern[]] := api[
 Module[{s, cutoff, parameter, coefficients, callback, settings, integers, reals, result},
  validateOptions[ComputeCoherentAmplitude, {opts}];
  s = integerVector[spins, 10, "spins"]; cutoff = nonnegativeInteger[Dl, "Dl"];
  parameter = finiteReal[gamma, "gamma"];
  coefficients = MapIndexed[complexVector[#1, "C" <> ToString[First[#2]]] &, {c1, c2, c3, c4, c5}];
  callback = OptionValue["ProgressFunction"]; settings = progressSettings[OptionValue["Backend"], callback];
  integers = Join[{cutoff, First[settings]}, s, Length /@ coefficients];
  reals = Prepend[Flatten[coefficients], parameter];
  result = checked[If[TrueQ[settings[[2]]], runJob[55, integers, reals, callback, settings[[3]]],
    callNative[16, integers, reals]]];
  complexValue[result]
 ]];
Options[ComputeFullTensor] = {"Backend" -> Automatic, "ProgressFunction" -> None, "ReturnType" -> "Tensor"};
ComputeFullTensor[spins_, Dl_, gamma_, opts : OptionsPattern[]] := api[
 Module[{s, cutoff, parameter, callback, settings, returnType, result},
  validateOptions[ComputeFullTensor, {opts}];
  s = integerVector[spins, 10, "spins"]; cutoff = nonnegativeInteger[Dl, "Dl"];
  parameter = finiteReal[gamma, "gamma"]; returnType = OptionValue["ReturnType"];
  enumIndex[returnType, {"Tensor", "Array"}, "ReturnType"];
  callback = OptionValue["ProgressFunction"]; settings = progressSettings[OptionValue["Backend"], callback];
  result = checked[withNewTensor[Function[id,
    If[TrueQ[settings[[2]]],
      runJob[50, Join[{id, cutoff, First[settings]}, s], {parameter}, callback, settings[[3]]],
      callNative[12, Join[{id, cutoff, First[settings]}, s], {parameter}]]]]];
  If[returnType === "Array",
    WithCleanup[Null, TensorArray[result], AbortProtect[discardTensorObject[result]]], result]
 ]];

DefaultConfig[arguments___] := failure["InvalidArgument", "DefaultConfig expects no arguments."];
CurrentConfig[arguments___] := failure["InvalidArgument", "CurrentConfig expects no arguments."];
Configure[arguments___] := failure["InvalidArgument", "Configure expects one complete configuration Association."];
LibraryVersion[arguments___] := failure["InvalidArgument", "LibraryVersion expects no arguments."];
BuildInfo[arguments___] := failure["InvalidArgument", "BuildInfo expects no arguments."];
BackendAvailableQ[arguments___] := failure["InvalidArgument", "BackendAvailableQ expects CPU or GPU as a string."];
CUDADeviceCount[arguments___] := failure["InvalidArgument", "CUDADeviceCount expects no arguments."];
CUDADeviceInfo[arguments___] := failure["InvalidArgument", "CUDADeviceInfo expects one optional zero-based ordinal."];
ComputeAmplitude[arguments___] := failure["InvalidArgument", "ComputeAmplitude expects spins, intertwiners, Dl, gamma, and optional rules."];
ComputeCoherentAmplitude[arguments___] := failure["InvalidArgument", "ComputeCoherentAmplitude expects spins, Dl, gamma, then angles or five coefficient vectors, and optional rules."];
ComputeFullTensor[arguments___] := failure["InvalidArgument", "ComputeFullTensor expects spins, Dl, gamma, and optional rules."];
ComputeCoherentCoefficients[arguments___] := failure["InvalidArgument", "ComputeCoherentCoefficients expects four spins and four angle pairs."];
CheckClosure[arguments___] := failure["InvalidArgument", "CheckClosure expects spins, angles, and optional tolerance rules."];
IndicesFromIntertwiners[arguments___] := failure["InvalidArgument", "IndicesFromIntertwiners expects spins and intertwiners."];

End[];
EndPackage[];
