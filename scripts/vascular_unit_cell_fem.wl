(* ::Package:: *)

(* =================================================================================================
   PERIODIC UNIT CELL  --  vascular spacing in the INTERIOR of a vascularised tumour.

   THE UNIT CELL. Vessels sit on a square lattice, so one s x s square centred on a vessel
   repeats indefinitely. By symmetry the midplanes between neighbouring vessels carry ZERO NORMAL
   FLUX, which is the natural boundary condition here, so no periodic wrapping is needed. The cell
   mean equals the infinite-lattice mean exactly:

        <NO>_cell  =  <NO>_infinite lattice

   REMOVED CONFOUNDS. Against a finite tumour disc this eliminates, simultaneously: the outer
   oxygen source, the changing vessel count, the vessel-wall-to-perimeter ratio, the distance from
   vessels to an external boundary, and the fraction of domain influenced by that boundary. The ONLY
   thing that varies across the sweep is the tissue assigned to each vessel, and vessel density goes
   as 1/s^2 -- which is the biological variable the figure claims to study.

   WHAT THIS MODEL IS. An idealised homogeneous region INSIDE a vascularised tumour, far
   enough from the tumour boundary that the local vascular microenvironment is periodic. It is NOT a
   finite tumour cross-section. That is a clean division of labour: Figure 4 owns the finite
   avascular spheroid with surface-supplied oxygen; this owns the vascularised interior.

   Each cell carries one vessel with all three functions -- O2 source, NOS3 source, luminal NO sink
   -- against uniform tumour NOS2, so the ratio of vascular processes to tumour volume rises as
   1/s^2 by construction rather than by argument.

   Usage: wolframscript -file vascular_unit_cell_fem.wl <spacing_um>
================================================================================================= *)

$HistoryLength = 0;
Needs["NDSolve`FEM`"];
here = DirectoryName[$InputFileName];
root = ParentDirectory[here];
cfg = Import[FileNameJoin[{root, "data", "fem_reference_constants.json"}], "RawJSON"];
get[k_] := Lookup[cfg, k, (Print["MISSING ", k]; Abort[])];

SP = If[Length[$ScriptCommandLine] >= 2, ToExpression[$ScriptCommandLine[[2]]], 150.];
DNO = get["D_NO_um2_per_s"]; DO2 = get["D_O2_um2_per_s"];
VM = get["V_M_uM_per_s"]; KMR = get["Km_resp_uM"]; KI = get["K_I_nM"];
VMAX2 = get["Vmax_NOS2_nM_per_s"]; KM2 = get["Km_NOS2_uM"]; PHI = get["phi"];
VMAX3 = get["Vmax_NOS3_nM_per_s"]; KM3 = get["Km_NOS3_uM"]; WSH = get["shell_um"];
AVES = get["vessel_radius_um"]; O2WALL = get["O2_wall_uM"];
FN2 = If[StringQ[Environment["FEM_F_NOS2"]], ToExpression[Environment["FEM_F_NOS2"]],
         get["f_nos2"]];
KLUM = If[StringQ[Environment["FEM_KLUMEN"]], ToExpression[Environment["FEM_KLUMEN"]],
          get["k_lumen_per_s"]];
ENOS = If[StringQ[Environment["FEM_ENOS"]], Environment["FEM_ENOS"] =!= "0", True];
NOFB = ! (StringQ[Environment["FEM_NO_FEEDBACK"]] &&
   Environment["FEM_NO_FEEDBACK"] === "0");
(* FEM_MAXCELL overrides the mesh element size, for the mesh-convergence gate ONLY.
   UNSET IS THE PRODUCTION PATH and must stay bit-identical: MCOVR = None leaves the
   original Min[6., H/12.] expression untouched below. The production mesh is SPACING-DEPENDENT
   (the 6 um cap stops binding between 150 and 100 um), so a sweep across spacing carries a mesh
   change alongside the physics -- which is what the gate exists to quantify. *)
MCOVR = If[StringQ[Environment["FEM_MAXCELL"]], ToExpression[Environment["FEM_MAXCELL"]], None];
KOBS = get["K_OBS"]; KANOX = get["K_ANOX"]; RHOC = get["RHO_CELLS"];

kdeg[c_] := KOBS*(c*10^-6)*RHOC + KANOX*RHOC;
prod2[c_] := VMAX2*c/(c + KM2)*PHI*FN2;
LL = Sqrt[DNO/KLUM];
hImp = DNO*BesselI[1, AVES/LL]/(LL*BesselI[0, AVES/LL]);
b = AVES + WSH;
H = SP/2.;

(* transcription gate, same values the finite model checks *)
chk = get["kdeg_check"];
If[Abs[kdeg[56.] - Lookup[chk, "56"]] > 1.*^-9,
  Print["\[UpTee] TRANSCRIPTION GATE FAILED"]; Abort[]];
Print["PERIODIC UNIT CELL  s = ", SP, " um   half-width ", H,
      "   a = ", AVES, "   shell ", WSH];
Print["  f_NOS2 ", FN2, "   eNOS ", If[ENOS, "ON", "OFF"], "   k_lumen ", KLUM,
      "   h ", ToString[NumberForm[hImp, 6]], " um/s"];
If[H <= b + 1.,
  Print["\[UpTee] CELL SMALLER THAN THE ENDOTHELIAL SHELL - spacing too small"]; Abort[]];

(* ---- domain: square minus the lumen. Zero normal flux on the four midplanes is NATURAL. ------ *)
cell = RegionDifference[Rectangle[{-H, -H}, {H, H}], Disk[{0., 0.}, AVES]];
MCUSED = If[NumericQ[MCOVR], N[MCOVR], Min[6., H/12.]];
mesh = ToElementMesh[cell, "MaxBoundaryCellMeasure" -> 0.3,
   MaxCellMeasure -> {"Length" -> MCUSED}, "MeshOrder" -> 2];
Print["  max cell ", MCUSED, " um", If[NumericQ[MCOVR], "  (FEM_MAXCELL OVERRIDE)", ""]];
Print["  mesh ", Length[mesh["Coordinates"]], " nodes"];

onWall[x_, y_] := x^2 + y^2 < (AVES + 0.4)^2;
inShell[x_?NumericQ, y_?NumericQ] := If[AVES^2 < x^2 + y^2 <= b^2, 1., 0.];

coords = mesh["Coordinates"];
o2V = ConstantArray[O2WALL, Length[coords]];
nrelV = ConstantArray[0., Length[coords]];
theta = 0.5; it = 0; dev = Infinity;

Do[
  (* FEM_NO_FEEDBACK=0 removes the NO-dependence of respiration, turning the competitive
     inhibition term off while leaving everything else identical. It exists to test whether the
     anoxia trend across NOS2 fraction is actually caused by NO sparing oxygen, or merely
     consistent with it. *)
  coefA = ElementMeshInterpolation[{mesh},
    VM/(Clip[o2V, {0., \[Infinity]}] + KMR*(1. + If[NOFB, nrelV/KI, 0.*nrelV]))];
  o2f = NDSolveValue[{
     Inactive[Div][-DO2*Inactive[Grad][u[x, y], {x, y}], {x, y}] + coefA[x, y]*u[x, y] == 0,
     DirichletCondition[u[x, y] == O2WALL, onWall[x, y]]},
    u, {x, y} \[Element] mesh,
    Method -> {"FiniteElement", "InterpolationOrder" -> {u -> 2}}];
  o2V = Clip[o2f @@@ coords, {0., \[Infinity]}];
  o2f = ElementMeshInterpolation[{mesh}, o2V];
  nof = NDSolveValue[{
     Inactive[Div][-DNO*Inactive[Grad][w[x, y], {x, y}], {x, y}]
       + kdeg[o2f[x, y]]*w[x, y] - prod2[o2f[x, y]]
       - If[TrueQ[ENOS], inShell[x, y]*VMAX3*o2f[x, y]/(o2f[x, y] + KM3), 0.] ==
       NeumannValue[-hImp*w[x, y], onWall[x, y]]},
    w, {x, y} \[Element] mesh,
    Method -> {"FiniteElement", "InterpolationOrder" -> {w -> 2}}];
  nrelV = Clip[theta*(nof @@@ coords) + (1. - theta)*nrelV, {0., \[Infinity]}];
  If[it > 0, dev = Max[Abs[o2V - o2prev]]];
  o2prev = o2V; it = i;
  If[NumericQ[dev] && dev < 1.*^-4, Break[]],
 {i, 60}];
If[! (NumericQ[dev] && dev < 1.*^-4),
  Print["\[UpTee] NOT CONVERGED (", dev, ") - no result"]; Abort[]];
Print["  converged in ", it, " coupling iterates"];

area = NIntegrate[1, {x, y} \[Element] mesh];
meanNO = NIntegrate[nof[x, y], {x, y} \[Element] mesh]/area;
meanO2 = NIntegrate[o2f[x, y], {x, y} \[Element] mesh]/area;
anox = NIntegrate[Boole[o2f[x, y] < 0.1], {x, y} \[Element] mesh]/area;
sig = NIntegrate[Boole[nof[x, y] > 1.], {x, y} \[Element] mesh]/area;
(* BRIDGING OBSERVABLES. The signalled fraction saturates at 100% for every spacing once
   NOS2 >= 5%, and is identically 0 at 0.5%, so it carries no information at either end of the
   abundance axis. The mean does not saturate but ignores the sGC threshold entirely. What sets
   coverage is the WORST-SERVED POINT: the cell corner, farthest from its vessel at distance
   H*Sqrt[2]. Coverage is essentially a step function of that one number, so it is the observable
   that stays informative across the whole grid.
   noCorner AND noMin ARE A CROSS-CHECK ON EACH OTHER, not redundant output. NO should fall
   monotonically away from the vessel, putting the true minimum AT the corner. If noMin sits
   materially below noCorner the field is not monotone and the bridging picture is wrong. *)
noCorner = nof[H, H];
(* EDGE MIDPOINT = the saddle between two NEIGHBOURING vessels, at (H, 0). Distinct from the
   corner (H, H), which sits between FOUR vessels. Nearest-neighbour connectivity and full-domain
   coverage are two different threshold events and need two different probes. *)
noEdge = nof[H, 0.];
noWall = nof[AVES, 0.];
(* THE MINIMUM IS MEASURED, NEVER ASSUMED TO BE THE CORNER. At 1% NOS2 the profile INVERTS
   between 250 and 150 um: at 250 um the corner is the minimum (oxygen-starved, NOS2 at a few percent
   of Vmax because Km is 42 uM), while at 150 um the corner is the MAXIMUM and the minimum sits at
   the vessel wall. SAY SINK-LIKE, NOT "NET SINK": that is the field's SPATIAL SIGNATURE, and
   which way the net flux runs is a separate question answered by the two integrals below. Record
   where the minimum actually is. *)
noVals = nof @@@ coords;
minIdx = First[Ordering[noVals, 1]];
minLoc = coords[[minIdx]];
noMin = noVals[[minIdx]];
minR = Norm[minLoc];
maxIdx = First[Ordering[noVals, -1]];
noMax = noVals[[maxIdx]];
maxR = Norm[coords[[maxIdx]]];

(* NET VESSEL BALANCE -- so "source" vs "sink" stops being an inference from the field's shape
   and becomes a measurement. Where the extrema sit says the vessel is ASSOCIATED WITH a local maximum
   or minimum; it does not establish which way the net flux runs. That needs the integrals compared:
     endothelial NOS3 production   \[Integral]_shell Vmax3 c/(c+Km3) dA
     luminal removal               \[ContourIntegral]_wall h w ds
   Both are per unit length of vessel and share units, so the difference is meaningful.

   NEVER A SAMPLED MEAN TIMES A LENGTH. The first version computed
       wallAvg = Mean of nof on a circle at r = AVES + 0.05;  removal = hImp*wallAvg*2 Pi AVES
   and left the global NO budget 1.72% open, IDENTICALLY at 6 um and 3 um mesh -- mesh-independent,
   so a fixed bias, not discretisation. Cause: the luminal Robin condition has its own length scale
   L = Sqrt[DNO/KLUM] ~ 1.28 um, giving a wall gradient of about 0.66 /um, so a 0.05 um offset
   overstates w by ~3.3% -- which is ~990 of the 1178 residual, with the right sign. Integrate ON
   the boundary instead. *)
(* GATE ON ENOS. Ungated, an eNOS-OFF run reports a nonzero NOS3 production that is not
   in the PDE at all -- the shell source term is switched out of the equation above but this
   integral would still evaluate it. Same class as the eNOS flag that once failed to select. *)
prodE = If[TrueQ[ENOS],
   NIntegrate[inShell[x, y]*VMAX3*o2f[x, y]/(o2f[x, y] + KM3), {x, y} \[Element] mesh], 0.];
removal = hImp*NIntegrate[nof[AVES*Cos[t], AVES*Sin[t]]*AVES, {t, 0., 2. Pi}];
wallAvg = removal/(hImp*2. Pi*AVES);        (* the true boundary mean, back-computed *)

(* FOUR DISTINCT WALL QUANTITIES, ANSWERING DIFFERENT QUESTIONS. Calling any of them simply
   "C_wall" is how a 0.56 nM break-even concentration got into circulation when the true value is
   0.389 nM.
     no_wall_point_nM  value at (a, 0). ORIENTATION-SPECIFIC. Kept only for continuity.
     no_wall_mean_nM   circumferential MEAN. The flux variable: Q_lumen = h <C> 2 pi a.
     no_wall_min_nM    circumferential MINIMUM. THE SIGNALLING-TOPOLOGY VARIABLE: min < 1 nM
                       means some vessel-adjacent tissue is subthreshold.
     no_wall_max_nM    circumferential MAXIMUM. max < 1 nM means the WHOLE circumference is
                       subthreshold; min < 1 < max is angularly heterogeneous wall signalling.
   The regime diagram's near-horizontal contour was built on the POINT value, which is an
   orientation-specific threshold, NOT the condition for a perivascular gap to exist. *)
wallRing = Table[nof[AVES*Cos[t], AVES*Sin[t]], {t, 0., 2. Pi - Pi/180., Pi/180.}];
wallMin = Min[wallRing];
wallMax = Max[wallRing];
wallHet = TrueQ[wallMin < 1. < wallMax];
Print["  wall NO: point ", ToString[NumberForm[nof[AVES, 0.], 5]],
      "   mean ", ToString[NumberForm[wallAvg, 5]],
      "   min ", ToString[NumberForm[wallMin, 5]],
      "   max ", ToString[NumberForm[wallMax, 5]],
      If[wallHet, "   ANGULARLY HETEROGENEOUS at 1 nM", ""]];

(* INDEPENDENT FLUX IDENTITY. The same efflux computed from the GRADIENT rather than from the
   Robin coefficient. Outward normal at the lumen is -rhat, so the flux leaving the tissue is
       J.n = -DNO Grad[w].(-rhat) = DNO dw/dr
   If these two disagree the boundary implementation is wrong; if they agree, it is
   exonerated and any residual lives in the volume terms. *)
removalFlux = NIntegrate[
   DNO*(Derivative[1, 0][nof][AVES*Cos[t], AVES*Sin[t]]*Cos[t]
      + Derivative[0, 1][nof][AVES*Cos[t], AVES*Sin[t]]*Sin[t])*AVES, {t, 0., 2. Pi}];
fluxDisc = Abs[removalFlux - removal]/Max[Abs[removal], 1.*^-12];
Print["  luminal efflux: Robin ", ToString[NumberForm[removal, 6]],
      "   gradient ", ToString[NumberForm[removalFlux, 6]],
      "   disagreement ", ToString[CForm[100. fluxDisc]], "%"];
If[fluxDisc > 0.02,
  Print["\[UpTee] THE TWO BOUNDARY ESTIMATES DISAGREE - boundary implementation suspect"]];

(* Q_vascular_balance is NOT the net flux exchanged across the whole vessel/endothelium-tissue structure, because it
   omits diffusive export across r = b and kdeg losses on shell-derived NO. What is actually
   measured is narrower and unambiguous, so say only that:
       Q_vascular_balance = Q_NOS3 - Q_lumen
   "Among the two explicitly vessel-associated NO terms, does endothelial production exceed
   or fall below luminal removal?" Diffusion is excluded because it redistributes NO WITHIN the
   domain; distributed degradation is excluded because it is a tissue-wide sink acting on the
   resulting field, not a vessel-associated term.
   REPORT IT AS "luminal removal exceeded endothelial NOS3 production", NEVER as "the vessel
   is a net sink" -- the latter reads as a total vessel-tissue exchange flux, which this is not. *)
qVasc = prodE - removal;
(* WHAT netVes IS, AND WHAT IT IS NOT. It is NOT the net flux the shell exchanges with surrounding tissue,
   because it omits diffusive export across r = b and local kdeg losses on shell-derived NO.
   What it IS: the vessel structure's contribution to the CELL-WIDE NO budget -- what the
   vessel adds (endothelial production) minus what it removes (luminal efflux). Diffusion is
   internal redistribution within the cell, not a loss from it, and kdeg is a distributed tissue
   sink acting on NO of any origin, so neither belongs in a vessel term. If instead you want the
   shell's net export to tissue, that is a DIFFERENT quantity (flux at r = b) and this is the wrong
   number for it. Never quote netVes for that question. *)

(* GLOBAL NO BUDGET GATE. At steady state the whole cell must satisfy
       total production  =  total degradation  +  luminal efflux
   This closes the accounting: it validates the NO solve AND confirms that prodE and removal are on
   the same footing as the other terms. A budget that does not close means the flux
   bookkeeping above is wrong, whatever the field looks like. *)
totProd = NIntegrate[prod2[o2f[x, y]]
    + If[TrueQ[ENOS], inShell[x, y]*VMAX3*o2f[x, y]/(o2f[x, y] + KM3), 0.],
   {x, y} \[Element] mesh];
totDeg = NIntegrate[kdeg[o2f[x, y]]*nof[x, y], {x, y} \[Element] mesh];
budRes = totProd - totDeg - removal;
budRel = Abs[budRes]/Max[totProd, 1.*^-12];
Print["  NO budget: production ", ToString[NumberForm[totProd, 6]],
      " = degradation ", ToString[NumberForm[totDeg, 6]],
      " + luminal ", ToString[NumberForm[removal, 6]],
      "   residual ", ToString[CForm[100. budRes/Max[totProd, 1.*^-12]]],
      "% (signed)  ", ToString[CForm[100. budRel]], "% (abs)  raw ",
      ToString[CForm[budRes]]];
If[budRel > 0.02,
  Print["\[UpTee] NO BUDGET DOES NOT CLOSE (", ToString[NumberForm[100. budRel, 3]],
        "%) - flux bookkeeping is wrong"]];

(* RESIDUALS ALWAYS IN CForm. ScientificForm TYPESETS IN 2D and ToString of it splits the
   line, which is how a -2.5*10^-4 % residual printed as a bare "-8". NumberForm rendered a
   -2.5*10^-4 % residual as a bare "-8" in the log, which reads as a catastrophic budget
   failure rather than a clean close. A diagnostic whose formatting can be mistaken for the
   defect it screens for is worse than no diagnostic. *)

(* =============================================================================================
   TAGGED-SOURCE eNOS FATE DECOMPOSITION.  Enable with FEM_TAG_ENOS=1.

   THE eNOS ON/OFF DIFFERENCE IS NOT THIS. Differencing Q_lumen between eNOS-on and
   eNOS-off runs measures the RESPONSE of luminal removal to adding NOS3, not the FATE of
   NOS3-derived NO. Switching eNOS on also raises NO, which inhibits respiration, which raises O2,
   which raises NOS2 production -- so that difference carries secondary NOS2-derived NO with it.
   The 86.7% measured that way says "adding NOS3 raises luminal removal by 86.7% of NOS3
   production", which is a DIFFERENT SENTENCE from "86.7% of NOS3-derived NO is recaptured".

   WHY A TAGGED SOLVE IS EXACT. At FROZEN O2 the NO operator is LINEAR in w:
       -D Lap w + kdeg(O2) w = S,   Robin at the wall
   so superposition holds exactly. Solving that same operator with ONLY the shell source gives a
   genuinely source-attributed field, and its luminal integral is true recapture.

   THE FATES MUST SUM TO PRODUCTION -- that is the gate:
       Q_NOS3  =  Q_lumen_NOS3 (recaptured)  +  Q_deg_NOS3 (distributed degradation)
   THE COMPLEMENT OF RECAPTURE IS NOT DELIVERY. Most of what is not recaptured is
   degraded in tissue. Delivery past the endothelium is a separate quantity: the outward flux
   across r = b. Never report 1 - f_recapture as "reaches tissue".
============================================================================================= *)
If[TrueQ[ENOS] && StringQ[Environment["FEM_TAG_ENOS"]] &&
   Environment["FEM_TAG_ENOS"] =!= "0",
 Module[{tag, qLum, qDeg, qOut, tagRes},
  tag = NDSolveValue[{
     Inactive[Div][-DNO*Inactive[Grad][v[x, y], {x, y}], {x, y}]
       + kdeg[o2f[x, y]]*v[x, y]
       - inShell[x, y]*VMAX3*o2f[x, y]/(o2f[x, y] + KM3) ==
       NeumannValue[-hImp*v[x, y], onWall[x, y]]},
    v, {x, y} \[Element] mesh,
    Method -> {"FiniteElement", "InterpolationOrder" -> {v -> 2}}];
  qLum = hImp*NIntegrate[tag[AVES*Cos[t], AVES*Sin[t]]*AVES, {t, 0., 2. Pi}];
  qDeg = NIntegrate[kdeg[o2f[x, y]]*tag[x, y], {x, y} \[Element] mesh];
  qOut = NIntegrate[-DNO*(Derivative[1, 0][tag][b*Cos[t], b*Sin[t]]*Cos[t]
       + Derivative[0, 1][tag][b*Cos[t], b*Sin[t]]*Sin[t])*b, {t, 0., 2. Pi}];
  tagRes = (prodE - qLum - qDeg)/Max[prodE, 1.*^-12];
  Print["  TAGGED eNOS: production ", ToString[NumberForm[prodE, 6]],
        " = luminal ", ToString[NumberForm[qLum, 6]],
        " + degraded ", ToString[NumberForm[qDeg, 6]],
        "   residual ", ToString[CForm[100. tagRes]], "%  raw ",
        ToString[CForm[prodE - qLum - qDeg]]];
  Print["    f_recapture ", ToString[NumberForm[100. qLum/Max[prodE, 1.*^-12], 4]],
        "%   f_degraded ", ToString[NumberForm[100. qDeg/Max[prodE, 1.*^-12], 4]],
        "%   outward flux at r=b ", ToString[NumberForm[qOut, 6]],
        " (", ToString[NumberForm[100. qOut/Max[prodE, 1.*^-12], 4]], "% of production)"];
  If[Abs[tagRes] > 0.02,
    Print["\[UpTee] TAGGED BUDGET DOES NOT CLOSE - the tagged solve is not trustworthy"]]]];

Print["  NOS3 production ", ToString[NumberForm[prodE, 6]],
      "   luminal removal ", ToString[NumberForm[removal, 6]],
      "   balance ", ToString[NumberForm[qVasc, 6]],
      If[qVasc < 0., "  -> removal exceeds NOS3 production", "  -> NOS3 production exceeds removal"]];
(* the shell integral is the startup gate: source present, right place, right total *)
shellA = NIntegrate[inShell[x, y], {x, y} \[Element] mesh];
shellX = Pi*(b^2 - AVES^2);
Print["  endothelial area ", ToString[NumberForm[shellA, 6]], " vs analytic ",
      ToString[NumberForm[shellX, 6]], "  err ",
      ToString[NumberForm[100. Abs[shellA - shellX]/shellX, 4]], "%"];
Print["  tissue area ", ToString[NumberForm[area, 7]], " um^2   (s^2 - pi a^2 = ",
      ToString[NumberForm[SP^2 - Pi AVES^2, 7]], ")"];
Print["  mean NO ", ToString[NumberForm[meanNO, 6]], " nM   mean O2 ",
      ToString[NumberForm[meanO2, 6]], " uM   anoxic ", ToString[NumberForm[100. anox, 4]],
      "%   NO>1nM ", ToString[NumberForm[100. sig, 5]], "%"];

(* --- optional field dump for Panel B: the cell is TILED by the renderer, so one cell is all
   that is ever needed. Gated on FEM_DUMP_FIELD so the sweep runs stay fast. *)
If[StringQ[Environment["FEM_DUMP_FIELD"]] && Environment["FEM_DUMP_FIELD"] =!= "0",
 Module[{ng = 160, ax2, no2, o22, safe, want, got},
  ax2 = Table[x, {x, -H, H, 2. H/(ng - 1)}];
  safe[f_, xx_, yy_, r_] := Module[{v}, v = Quiet@Check[f[xx, yy], $Failed];
     If[NumericQ[v], Round[v, r], Null]];
  (* the lumen is not in the domain; sampling it returns Indeterminate and breaks the
     JSON serialiser, which is how the finite model produced zero-byte field files. *)
  no2 = Table[If[xx^2 + yy^2 > (AVES + 0.3)^2, safe[nof, xx, yy, 0.0001], Null],
     {yy, ax2}, {xx, ax2}];
  o22 = Table[If[xx^2 + yy^2 > (AVES + 0.3)^2, safe[o2f, xx, yy, 0.001], Null],
     {yy, ax2}, {xx, ax2}];
  want = Count[Flatten[Table[If[xx^2 + yy^2 > (AVES + 0.3)^2, 1, 0], {yy, ax2}, {xx, ax2}]], 1];
  got = Count[Flatten[no2], _?NumericQ];
  Print["  field ", got, " of ", want, " cells (",
        ToString[NumberForm[100. got/Max[want, 1], 4]], "%)"];
  If[got < 0.95 want, Print["  \[UpTee] FIELD COVERAGE TOO LOW - not exporting"],
   Export[FileNameJoin[{root, "data", "cellfield_" <> ToString[Round[SP]] <> ".json"}],
    <|"model" -> "periodic_unit_cell", "spacing_um" -> SP, "half_um" -> H,
      "vessel_radius_um" -> AVES, "mean_NO_nM" -> meanNO, "mean_O2_uM" -> meanO2,
      "axis_um" -> ax2, "NO_nM" -> no2, "O2_uM" -> o22|>, "JSON"];
   Module[{fp = FileNameJoin[{root, "data", "cellfield_" <> ToString[Round[SP]] <> ".json"}]},
     Print["  field dumped: ", FileNameTake[fp], "  ",
           Round[FileByteCount[fp]/1024], " kB"]]]]];

Export[FileNameJoin[{root, "data", "cell_" <> ToString[Round[SP]] <>
    "_f" <> ToString[FN2] <> "_enos" <> If[TrueQ[ENOS], "1", "0"] <>
    "_k" <> ToString[Round[KLUM]] <> If[NOFB, "", "_nofb"] <>
    (* WITHOUT THIS A GATE RUN OVERWRITES PRODUCTION. The name carried no mesh,
       so re-solving one (spacing, fraction) at a different element size would land on the
       SAME file and silently replace a production result with a diagnostic one. *)
        (* integer centi-microns: ToString[6.0] is "6." and would give "_m6..json" *)
    If[NumericQ[MCOVR], "_m" <> ToString[Round[MCUSED*100]], ""] <> ".json"}],
 <|"model" -> "periodic_unit_cell", "spacing_um" -> SP, "f_nos2" -> FN2, "enos" -> ENOS,
   "k_lumen_per_s" -> KLUM, "no_feedback" -> NOFB, "mesh_nodes" -> Length[coords],
   "max_cell_um" -> MCUSED, "mesh_overridden" -> NumericQ[MCOVR], "iterates" -> it,
   "tissue_area_um2" -> area, "mean_NO_nM" -> meanNO, "mean_O2_uM" -> meanO2,
   "anoxic_frac" -> anox, "sig_frac" -> sig,
   "no_corner_nM" -> noCorner, "no_edge_mid_nM" -> noEdge, "no_wall_nM" -> noWall,
   "no_min_nM" -> noMin, "no_min_r_um" -> minR, "no_max_nM" -> noMax, "no_max_r_um" -> maxR,
   "nos3_production" -> prodE, "luminal_removal" -> removal, "q_vascular_balance" -> qVasc, "luminal_removal_flux" -> removalFlux,
   (* the boundary MEAN of w -- a fourth distinct "wall" quantity, and the one the balance
      uses. no_wall_nM is a single point at (a,0) and reads ~4% higher because the field
      varies around the wall toward the edge midpoint vs the corner. *)
   "no_wall_point_nM" -> nof[AVES, 0.], "no_wall_mean_nM" -> wallAvg,
   "no_wall_min_nM" -> wallMin, "no_wall_max_nM" -> wallMax,
   "wall_angularly_heterogeneous" -> wallHet, "wall_mean_nM" -> wallAvg, "flux_disagreement" -> fluxDisc,
   "budget_residual_rel" -> (budRes/Max[totProd, 1.*^-12]),
   "vessel_is_net_sink" -> (qVasc < 0.),
   "corner_is_min" -> (Abs[noMin - noCorner] <= 0.02*Max[noCorner, 1.*^-9]),
   "profile_inverted" -> (noCorner > noWall), "shell_area_err" -> Abs[shellA - shellX]/shellX|>,
 "JSON"];
Print["done"];
