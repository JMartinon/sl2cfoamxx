(* ::Package:: *)

(* SPDX-FileCopyrightText: 2026 Jeremy Martinon *)
(* SPDX-License-Identifier: GPL-3.0-or-later *)

(* A separate direct-kernel invocation catches crashes during unload or exit. *)
If[Environment["SL2CFOAMXX_MATHEMATICA_TEST_MODE"] === "lifecycle",
    package = Environment["SL2CFOAMXX_MATHEMATICA_PACKAGE_DIR"];
    entry = FileNameJoin[{package,"Kernel","init.m"}];
    library = FileNameJoin[{package,"LibraryResources",$SystemID,
        "libsl2cfoamxx_mathematica" <> If[$OperatingSystem === "MacOSX",".dylib",".so"]}];
    directory = CreateDirectory[];
    checks = 0;
    SetAttributes[require,HoldFirst];
    require[condition_,message_] := If[TrueQ[condition],checks++,Print["FAIL: ",message];Exit[1]];

    WithCleanup[Null,
        Do[
            Get[entry];
            require[StringQ[SL2CfoamXX`LibraryVersion[]],"load and version query"];
            config = SL2CfoamXX`DefaultConfig[];
            config["Backend"] = "CPU";
            config["CPUThreads"] = 1;
            config["Tensors"] = Join[config["Tensors"],<|"Directory" -> directory,
                "Storage" -> "in_memory_dense","ReuseExisting" -> False|>];
            require[!FailureQ[SL2CfoamXX`Configure[config]],"configure after load"];
            tensor = SL2CfoamXX`ComputeFullTensor[ConstantArray[1,10],1,1.2];
            require[MatchQ[tensor,_SL2CfoamXX`FullTensorObject],"compute after load"];
            array = Normal[tensor];
            require[Dimensions[array] === {2,2,2,2,2},"packed array before unload"];
            reference = array[[1,1,1,1,1]];
            require[MachineNumberQ[reference],"native scalar"];
            require[LibraryUnload[library] === Null,"explicit LibraryLink unload"];
            require[array[[1,1,1,1,1]] === reference,"owned Wolfram array survives unload"];
            Clear[tensor,array];
            Print["Completed native unload/reload cycle ",cycle],
            {cycle,3}];
        Print["Mathematica lifecycle: ",checks," passed"],
        Quiet[DeleteDirectory[directory,DeleteContents -> True]]
    ];
    Exit[0];
];

(* Licensed-kernel integration test; GPU absence is an explicit skip (77). *)
Get[FileNameJoin[{Environment["SL2CFOAMXX_MATHEMATICA_PACKAGE_DIR"], "Kernel", "init.m"}]];
backend = Replace[Environment["SL2CFOAMXX_MATHEMATICA_TEST_BACKEND"], Except["GPU"] -> "CPU"];
available = BackendAvailableQ[backend];
If[FailureQ[available], Print[available]; Exit[1]];
If[!TrueQ[available], Print["SKIP: requested Mathematica GPU backend unavailable"]; Exit[77]];

$checks = 0; $failures = 0;
SetAttributes[assert, HoldRest];
assert[name_String, condition_] := If[TrueQ[condition], $checks++,
    $failures++; Print["FAIL: ", name, " -- ", HoldForm[condition]]];
close[a_, b_, tolerance_:5.*^-11] := TrueQ[NumericQ[a] && NumericQ[b] &&
    Abs[a-b] <= tolerance Max[Abs[b],10.^-300]];
arrayClose[a_, b_] := TrueQ[Dimensions[a] === Dimensions[b] &&
    Max[Abs[Flatten[a-b]]] <= 5.*^-11 Max[Abs[Flatten[b]]]];

spins = {2,1,1,2,1,1,2,1,1,1};
intertwiners = {3,2,0,2,2};
angles = {{.31,.13},{.77,-.42},{1.29,1.07},{2.11,2.21},
    {2.39,2.48},{1.47,-1.14},{.43,-.27},{.91,.61},
    {1.66,1.52},{2.52,-2.31},{.58,.36},{1.04,-.83},
    {1.18,.94},{1.83,-1.73},{2.67,2.69},{.69,-.48},
    {.82,.57},{1.96,1.89},{2.78,-2.52},{1.33,-1.06}};
nodeSpins = {{1,2,3,4},{5,6,7,1},{8,9,2,5},{10,3,6,8},{4,7,9,10}};
nodeRows = {{1,2,3,4},{6,7,8,5},{11,12,9,10},{16,13,14,15},{17,18,19,20}};
notebook = Get[FileNameJoin[{DirectoryName[$InputFileName],"..","examples","mathematica_example.nb"}]];
notebookInputs = Cases[notebook, Cell[BoxData[boxes_],"Input",___] :> boxes, Infinity];
notebookExpressions = Quiet[Check[ToExpression[#,StandardForm,HoldComplete],$Failed]] & /@ notebookInputs;
assert["notebook input syntax", Head[notebook] === Notebook &&
    Length[notebookExpressions] > 0 &&
    AllTrue[notebookExpressions,MatchQ[#,_HoldComplete] &]];
notebookSetup = Cases[notebook,
    Cell[BoxData[boxes_],"Input",InitializationCell -> True] :>
        Quiet[Check[ToExpression[boxes,StandardForm,HoldComplete],$Failed]], Infinity];
assert["notebook initialization syntax", Length[notebookSetup] > 0 &&
    AllTrue[notebookSetup,MatchQ[#,_HoldComplete] &]];
coefficientVectors[s_] := Table[ComputeCoherentCoefficients[s[[nodeSpins[[n]]]],
    angles[[nodeRows[[n]]]]],{n,5}];
explicitContraction[a_, cs_] := Total[Flatten[Array[
    Function[{i,j,k,l,m}, a[[i,j,k,l,m]] cs[[1,i]] cs[[2,j]] cs[[3,k]] cs[[4,l]] cs[[5,m]]],
    Dimensions[a]]]];

directory = CreateDirectory[];
WithCleanup[Null,
    defaults = DefaultConfig[];
    assert["default association", AssociationQ[defaults]];
    If[!AssociationQ[defaults], Print[defaults]; Exit[1]];
    assert["native version", LibraryVersion[] === BuildInfo[]["Version"]];
    assert["version value", LibraryVersion[] === "1.0.2"];
    lorentzian = $LorentzianBoundaryAngles;
    euclidean = $EuclideanBoundaryAngles;
    lorentzianSpins = {10,10,10,10,4,4,4,4,4,4};
    assert["preset shapes", Dimensions[lorentzian] === {20,2} && Dimensions[euclidean] === {20,2}];
    assert["packed real presets", Developer`PackedArrayQ[lorentzian] && Developer`PackedArrayQ[euclidean] &&
        MatrixQ[lorentzian,Head[#] === Real &] && MatrixQ[euclidean,Head[#] === Real &]];
    assert["Lorentzian preset row order", lorentzian[[{1,8,12,16,20}]] === {
        {1.570796326794897,0.000000000000000},
        {2.069985204145935,-0.320246032512208},
        {2.069985204145935,2.230879268761226},
        {1.201165252127538,-2.186276035465284},
        {1.940427401462255,-2.186276035465284}}];
    assert["Euclidean preset row order", euclidean[[{1,8,12,16,20}]] === {
        {0.000000000000000,0.000000000000000},
        {1.230959417340775,2.365313622849416},
        {0.729727656226966,-0.388139515370189},
        {2.411864997362827,0.659058035826409},
        {0.729727656226966,-2.482534617763384}}];
    modifiedPreset = euclidean; modifiedPreset[[1,1]] = -1.;
    assert["preset copy isolation", euclidean[[1,1]] === 0. && $EuclideanBoundaryAngles === euclidean];
    assert["Lorentzian preset closure", CheckClosure[lorentzianSpins,lorentzian]["AllClosed"]];
    assert["scaled Lorentzian preset closure", CheckClosure[2 lorentzianSpins,lorentzian]["AllClosed"]];
    assert["Euclidean preset closure", CheckClosure[ConstantArray[4,10],euclidean]["AllClosed"]];
    assert["presets require matching spin geometry", !CheckClosure[ConstantArray[4,10],lorentzian]["AllClosed"]];
    assert["native compiler", StringQ[BuildInfo[]["CompilerID"]]];
    assert["CPU available", BackendAvailableQ["CPU"]];
    assert["pattern enum rejected", FailureQ[BackendAvailableQ[_]]];
    assert["non-string enum rejected", FailureQ[BackendAvailableQ[0]]];
    If[backend === "GPU", assert["visible GPU", CUDADeviceCount[] > 0];
        assert["device name", StringQ[CUDADeviceInfo[0]["Name"]]]];
    assert["progress default", defaults["ProgressBar"] === True];
    assert["storage default", defaults["Tensors"]["Storage"] === "on_disk_sym"];
    assert["CPU threads default", defaults["CPUThreads"] === 0];
    assert["payload default", defaults["Tensors"]["MaximumPayloadSizeBytes"] === 107374182400];
    assert["simplicity default", defaults["SimplicityConstraint"] === "gamma_j"];
    config = defaults;
    config["Backend"] = backend; config["CPUThreads"] = 1;
    config["Verbosity"] = "silent"; config["ProgressBar"] = False;
    config["Tensors"] = Join[config["Tensors"], <|"Directory" -> directory,
        "Storage" -> "in_memory_dense", "ReuseExisting" -> False|>];
    config["CoherentAmplitude"] = Join[config["CoherentAmplitude"], <|"PruningRatio" -> 0.|>];
    assert["configure", !FailureQ[Configure[config]]];
    presetConfig = config; presetConfig["SimplicityConstraint"] = "gamma_j_p1";
    Configure[presetConfig];
    assert["Lorentzian preset amplitude reference", close[
        ComputeCoherentAmplitude[lorentzianSpins,2,1.2,lorentzian,"Backend" -> backend],
        -1.2613917379855847*^-17 - 1.039173505859162*^-16 I,5.*^-9]];
    Configure[config];
    snapshot = CurrentConfig[];
    snapshot["LowLevel"] = Join[snapshot["LowLevel"], <|"BoosterRhoTile" -> 8,
        "FullTensorCPUPBatch" -> 128, "FullTensorCPUFinalTile" -> "tile_4x8"|>];
    assert["tuning configure", !FailureQ[Configure[snapshot]]];
    snapshot["LowLevel"] = Join[snapshot["LowLevel"], <|"BoosterRhoTile" -> 4|>];
    assert["snapshot isolation", CurrentConfig[]["LowLevel"]["BoosterRhoTile"] === 8];
    snapshot["LowLevel"] = Join[snapshot["LowLevel"], <|"BoosterRhoTile" -> 3|>];
    assert["invalid configuration", FailureQ[Configure[snapshot]]];
    assert["invalid configure atomicity", CurrentConfig[]["LowLevel"]["BoosterRhoTile"] === 8];
    assert["unknown configuration key", FailureQ[Configure[Append[config,"Typo" -> 1]]]];
    snapshot = config; snapshot["ProgressBar"] = 1;
    assert["non-boolean config rejected", FailureQ[Configure[snapshot]]];
    Configure[config];
    assert["one-based indices", IndicesFromIntertwiners[spins, intertwiners] === {2,2,1,2,2}];

    cs = coefficientVectors[spins];
    assert["packed coefficients", Developer`PackedArrayQ[cs[[2]]]];
    assert["node2 independent coefficients", Max[Abs[cs[[2]] -
        {0.1282584174511503+0.1343652056345688 I,0.2484734822500689+0.06033324720018829 I}]] < 2.*^-13];
    assert["node5 independent coefficients", Max[Abs[cs[[5]] -
        {0.02331902555825567-0.1698302920514597 I,0.002828506333065372-0.2918289545554249 I}]] < 2.*^-13];
    closure = CheckClosure[spins, ConstantArray[0,{20,2}]];
    assert["closure false", closure["AllClosed"] === False];
    Do[scale = Total[spins[[nodeSpins[[n]]]]]/2;
        assert["closure physical spin weights", closure["Nodes"][[n]]["Vector"] == {0,0,scale}];
        assert["closure scale", closure["Nodes"][[n]]["Scale"] == scale], {n,5}];
    assert["zero closure", CheckClosure[ConstantArray[0,10],ConstantArray[0,{20,2}]]["AllClosed"]];

    Do[
        config["SimplicityConstraint"] = convention; Configure[config];
        cs = coefficientVectors[s];
        tensor = ComputeFullTensor[s, cutoff, 1.2, "Backend" -> backend];
        assert["managed computed tensor", MatchQ[tensor,_FullTensorObject]];
        If[FailureQ[tensor], Print[tensor]; Exit[1]];
        array = Normal[tensor];
        assert["packed canonical array", Developer`PackedArrayQ[array]];
        assert["logical dimensions", Dimensions[tensor] === If[s === spins,{2,2,2,2,2},{2,2,2,1,2}]];
        assert["logical rank", ArrayDepth[array] === 5];
        expected = explicitContraction[array,cs];
        assert["Dot tensor contraction", close[Fold[Dot,array,Reverse[cs]],expected]];
        coherent = ComputeCoherentAmplitude[s,cutoff,1.2,angles,"Backend" -> backend];
        assert["coherent/tensor agreement", close[coherent,expected]];
        supplied = ComputeCoherentAmplitude[s,cutoff,1.2,Sequence @@ cs];
        assert["supplied coefficients configured backend", close[supplied,expected]];
        custom = Table[Table[(n + k I)/(n+k),{k,Length[cs[[n]]]}],{n,5}];
        customExpected = explicitContraction[array,custom];
        assert["custom coefficients without conjugation or normalization", close[
            ComputeCoherentAmplitude[s,cutoff,1.2,Sequence @@ custom,"Backend" -> backend],customExpected]];
        If[backend === "GPU",
            assert["CPU/GPU coherent agreement", close[
                ComputeCoherentAmplitude[s,cutoff,1.2,angles,"Backend" -> "CPU"],expected]];
            assert["CPU/GPU supplied coefficient agreement", close[
                ComputeCoherentAmplitude[s,cutoff,1.2,Sequence @@ custom,"Backend" -> "CPU"],customExpected]]];
        If[s === spins,
            fixed = ComputeAmplitude[s,intertwiners,cutoff,1.2];
            assert["fixed/tensor agreement", close[fixed,TensorValue[tensor,{2,2,1,2,2}]]]];
        metadata = TensorMetadata[tensor];
        assert["scientific labels", metadata["Scientific"]["Spins"] === s];
        assert["scientific cutoff", metadata["Scientific"]["Dl"] === cutoff];
        assert["logical metadata", metadata["LogicalValues"] === Times @@ Dimensions[array]];
        config["Tensors"] = Join[config["Tensors"],<|"Storage" -> "in_memory_sym"|>];
        Configure[config]; compact = ComputeFullTensor[s,cutoff,1.2];
        assert["dense/compact equality", arrayClose[Normal[compact],array]];
        assert["physical value count", Length[StoredTensorValues[compact]] === TensorMetadata[compact]["StoredValues"]];
        If[s === spins, assert["reflection compact", TensorMetadata[compact]["StoredValues"] === 20]];
        assert["direct array return", arrayClose[ComputeFullTensor[s,cutoff,1.2,"ReturnType" -> "Array"],array]];
        config["Tensors"] = Join[config["Tensors"],<|"Storage" -> "in_memory_dense"|>];
        Configure[config],
        {s,{spins,{2,1,3,2,1,1,2,1,1,1}}}, {convention,{"gamma_j","gamma_j_p1"}}, {cutoff,0,2}];

    (* Every axis has identifiable, nonuniform entries: catches rank/order swaps. *)
    expected = N[ArrayReshape[Range[120],{2,3,4,1,5}]];
    tensor = TensorFromArray[expected];
    assert["all-entry row/column ordering", Normal[tensor] === expected];
    assert["asymmetric canonical get", TensorValue[tensor,{2,3,4,1,5}] == 120.];
    alias = tensor; SetTensorValue[alias,{1,2,3,1,4},987.];
    assert["managed aliases share", TensorValue[tensor,{1,2,3,1,4}] == 987.];
    array = Normal[tensor]; array[[1,2,3,1,4]] = -99.;
    assert["array copies independent", TensorValue[tensor,{1,2,3,1,4}] == 987.];
    cloned = CloneTensor[tensor]; SetTensorValue[cloned,{1,1,1,1,1}, -41.];
    assert["native clones independent", TensorValue[tensor,{1,1,1,1,1}] == 1.];
    descendant = Module[{parent = CloneTensor[tensor]}, {parent}];
    Clear[cloned,alias];
    assert["managed lifetime through containers", Normal[First[descendant]] === Normal[tensor]];
    assert["zero index rejected", FailureQ[TensorValue[tensor,{0,1,1,1,1}]]];
    assert["large index rejected", FailureQ[TensorValue[tensor,{3,1,1,1,1}]]];
    file = FileNameJoin[{directory,"mixed tensor \[Alpha].sl2cxx"}];
    assert["save", !FailureQ[SaveTensor[tensor,file]]];
    assert["load exact", Normal[LoadTensor[file]] === Normal[tensor]];
    assert["file metadata", TensorMetadata[file]["Shape"] === {2,3,4,1,5}];
    assert["Unicode path roundtrip", TensorMetadata[file]["Path"] === file];
    mapped = MapTensor[file];
    assert["readonly map", TensorMetadata[mapped]["Writable"] === False];
    assert["readonly rejection", FailureQ[SetTensorValue[mapped,{1,1,1,1,1},5.]]];
    assert["readonly canonical export", Normal[mapped] === Normal[tensor]];
    writable = MapTensor[file,"Access" -> "read_write"];
    SetTensorValue[writable,{1,1,1,1,1},37.]; FlushTensor[writable];
    assert["mapped persistence", TensorValue[LoadTensor[file],{1,1,1,1,1}] == 37.];
    created = CreateMappedTensor[FileNameJoin[{directory,"new.sl2cxx"}],{2,1,1,1,1}];
    assert["mapped creation", Normal[created] === ConstantArray[0.,{2,1,1,1,1}]];
    assert["tag ByteArray", Head[TensorTag[tensor]] === ByteArray];
    assert["tag length", Length[Normal[TensorTag[tensor]]] === 1024];
    assert["diagnostic format", StringQ[FormatTensor[tensor,"MaxValues" -> 3]]];
    assert["generic creation", Normal[CreateTensor[{2,1,1,1,1}]] === ConstantArray[0.,{2,1,1,1,1}]];

    config["Tensors"] = Join[config["Tensors"],<|"Storage" -> "on_disk_sym",
        "OutputPath" -> FileNameJoin[{directory,"scientific.sl2cxx"}]|>];
    Configure[config]; tensor = ComputeFullTensor[spins,1,1.2];
    converted = ConvertTensorStorage[tensor,"on_disk_dense",FileNameJoin[{directory,"dense.sl2cxx"}]];
    assert["storage conversion exact", Normal[converted] === Normal[tensor]];
    assert["dense owned copy", Normal[DenseCopyTensor[tensor]] === Normal[tensor]];
    config["Tensors"] = Join[config["Tensors"],<|"ReuseExisting" -> True|>]; Configure[config];
    events = {}; reused = ComputeFullTensor[spins,1,1.2,"Backend" -> backend,
        "ProgressFunction" -> (AppendTo[events,#] &)];
    assert["cache hit", Normal[reused] === Normal[tensor] && events === {}];
    assert["fixed cache reuse", close[ComputeAmplitude[spins,intertwiners,1,1.2,
        "CachePolicy" -> "prefer_full_tensor"],TensorValue[tensor,{2,2,1,2,2}]]];
    config["Tensors"] = Join[config["Tensors"],<|"Storage" -> "in_memory_dense",
        "OutputPath" -> None, "ReuseExisting" -> False|>]; Configure[config];

    Do[assert["invalid doubled labels", FailureQ[ComputeFullTensor[invalid,0,1.2]]],
        {invalid,{N[spins],ReplacePart[spins,1 -> -1],ConstantArray[True,10],Take[spins,9],ConstantArray[2^40,10]}}];
    Do[assert["invalid gamma", FailureQ[ComputeFullTensor[spins,0,invalid]]],
        {invalid,{Infinity,Indeterminate,1+I,True,"gamma",-1.2,-$MinMachineNumber}}];
    assert["negative gamma fixed", FailureQ[ComputeAmplitude[spins,intertwiners,0,-1.2]]];
    assert["negative gamma coherent", FailureQ[ComputeCoherentAmplitude[spins,0,-1.2,angles]]];
    assert["invalid cutoff", FailureQ[ComputeFullTensor[spins,-1,1.2]]];
    assert["invalid angles", FailureQ[ComputeCoherentAmplitude[spins,0,1.2,Transpose[angles]]]];
    cs = coefficientVectors[spins];
    Do[bad = ReplacePart[cs,n -> invalid];
        assert["invalid coefficient length or value", FailureQ[
            ComputeCoherentAmplitude[spins,0,1.2,Sequence @@ bad,"Backend" -> backend]]],
        {n,5}, {invalid,{{},{1}, {1,2,3}, {{1,2}}, {Infinity,1}, {1+I Infinity,1},
            {Indeterminate,1}, {True,1}, {"coefficient",1}}}];
    assert["negative gamma supplied coefficients", FailureQ[
        ComputeCoherentAmplitude[spins,0,-1.2,Sequence @@ cs]]];
    assert["coefficient unknown option", FailureQ[
        ComputeCoherentAmplitude[spins,0,1.2,Sequence @@ cs,"Typo" -> 1]]];
    assert["unknown option", FailureQ[ComputeFullTensor[spins,0,1.2,"Typo" -> 1]]];
    assert["arbitrary precision real input", NumericQ[ComputeAmplitude[spins,intertwiners,0,6/5]]];
    invalidfile = FileNameJoin[{directory,"invalid.sl2cxx"}]; Export[invalidfile,"invalid","Text"];
    failure = LoadTensor[invalidfile];
    assert["typed native error", FailureQ[failure] && failure["Code"] === "invalid_tensor_file" && failure["NativeCode"] === 3];
    assert["zero amplitude machine result", MachineNumberQ[ComputeAmplitude[ConstantArray[0,10],ConstantArray[0,5],0,1.2]]];

    events = {};
    tensor = ComputeFullTensor[spins,1,1.2,"Backend" -> backend,"ProgressFunction" -> (AppendTo[events,#] &)];
    assert["full progress starts", Length[events] >= 2 && First[events]["Status"] === "started"];
    assert["full progress ends", Last[events]["Status"] === "completed"];
    assert["progress counts", Last[events]["Completed"] === Last[events]["Total"]];
    assert["progress backend", AllTrue[events, #["Backend"] === backend &]];
    assert["progress operation", AllTrue[events, #["Operation"] === "full_tensor" &]];
    events = {}; ComputeCoherentAmplitude[spins,1,1.2,angles,"Backend" -> backend,
        "ProgressFunction" -> (AppendTo[events,#] &)];
    assert["coherent progress ends", Last[events]["Status"] === "completed"];
    assert["coherent progress operation", AllTrue[events, #["Operation"] === "coherent_amplitude" &]];
    events = {}; cs = coefficientVectors[spins];
    supplied = ComputeCoherentAmplitude[spins,1,1.2,Sequence @@ cs,"Backend" -> backend,
        "ProgressFunction" -> (AppendTo[events,#] &)];
    assert["supplied coefficient progress result", close[supplied,explicitContraction[Normal[tensor],cs]]];
    assert["supplied coefficient progress ends", Length[events] >= 2 && Last[events]["Status"] === "completed"];
    assert["supplied coefficient progress operation", AllTrue[events, #["Operation"] === "coherent_amplitude" &]];
    assert["coefficient callback backend required", FailureQ[
        ComputeCoherentAmplitude[spins,1,1.2,Sequence @@ cs,"ProgressFunction" -> Identity]]];
    assert["callback backend required", FailureQ[ComputeFullTensor[spins,1,1.2,"ProgressFunction" -> Identity]]];
    assert["non-function callback rejected", FailureQ[ComputeFullTensor[spins,1,1.2,
        "Backend" -> backend,"ProgressFunction" -> True]]];
    marker = Failure["CallbackFailure",<|"MessageTemplate" -> "test callback"|>];
    assert["coefficient callback Failure preserved", ComputeCoherentAmplitude[spins,1,1.2,
        Sequence @@ cs,"Backend" -> backend,"ProgressFunction" -> (marker &)] === marker];
    assert["callback Failure preserved", ComputeFullTensor[spins,1,1.2,"Backend" -> backend,
        "ProgressFunction" -> (marker &)] === marker];
    assert["callback Throw preserved", Catch[ComputeFullTensor[spins,1,1.2,"Backend" -> backend,
        "ProgressFunction" -> (Throw["escaped","test"] &)],"test"] === "escaped"];
    assert["callback Abort preserved", CheckAbort[ComputeFullTensor[spins,1,1.2,"Backend" -> backend,
        "ProgressFunction" -> (Abort[] &)],"aborted"] === "aborted"];
    assert["usable after callback errors", Normal[ComputeFullTensor[spins,1,1.2]] === Normal[tensor]];
    changed = config; changed["SimplicityConstraint"] = "gamma_j";
    captured = ComputeFullTensor[spins,1,1.2,"Backend" -> backend,
        "ProgressFunction" -> (Configure[changed] &)];
    assert["active configuration snapshot", Normal[captured] === Normal[tensor]];
    assert["new configuration applied", CurrentConfig[]["SimplicityConstraint"] === "gamma_j"];
    Configure[config];
    (* Exercise Wolfram's own packed BLAS after native computations, too. *)
    matrix = Developer`ToPackedArray[N[ArrayReshape[Range[400],{20,20}]]];
    assert["Wolfram BLAS coexistence", matrix . Transpose[matrix] ==
        Table[Sum[matrix[[i,k]] matrix[[j,k]],{k,20}],{i,20},{j,20}]];
    (* Load and exercise the example directly from the notebook. The package
       environment above takes precedence over NotebookDirectory in a kernel. *)
    If[AllTrue[notebookSetup,MatchQ[#,_HoldComplete] &],
        Scan[ReleaseHold,notebookSetup];
        assert["notebook uses Euclidean preset", $SL2CfoamXXExampleAngles === euclidean];
        notebookResult = RunSL2CfoamXXExample[backend];
        assert["self-contained notebook workflow", AssociationQ[notebookResult]];
        If[AssociationQ[notebookResult],
            assert["notebook packed array", Developer`PackedArrayQ[notebookResult["Array"]]];
            assert["notebook tensor shape", Dimensions[notebookResult["Array"]] === {2,2,2,2,2}];
            assert["notebook coefficient overload", close[notebookResult["CoefficientAmplitude"],notebookResult["CoherentAmplitude"]]];
            assert["notebook progress", Length[notebookResult["ProgressEvents"]] >= 4];
            assert["notebook cache cleanup", !DirectoryQ[CurrentConfig[]["Tensors"]["Directory"]]]
        ]
    ];
    Print["Mathematica ", backend, ": ", $checks, " passed, ", $failures, " failed"],
    Quiet[DeleteDirectory[directory,DeleteContents -> True]]
];
Exit[If[$failures == 0,0,1]];
