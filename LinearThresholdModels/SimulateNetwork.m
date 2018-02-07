function [sNetwork, ctfFinalAct, ctfFinalState, vfTotalInput, mfResponseTrace, mfStateTrace] = SimulateNetwork(sNetwork, fhMonitorFunction, tTimeStep, tSimDuration, fhInputFunction, bSaveMovie, vnSampleNeurons, strContinueBase)

% SimulateNetwork - FUNCTION Simulate an arbitrary network
%
% Usage: [sNetwork, ctfFinalAct, ctfFinalState, vfTotalInput, mfResponseTrace, mfStateTrace] = ...
%           SimulateNetwork(sNetwork, <fhMonitorFunction>, tTimeStep, tSimDuration, <fhInputFunction, bSaveMovie, vnSampleNeurons, strContinueBase>)

DEF_nSaveStatePeriod = 4000;

if (nargin < 4)
   disp('*** SimulateNetwork: Incorrect usage');
   help SimulateNetwork;
   return;
end

if (~exist('bSaveMovie', 'var'))
   bSaveMovie = false;
else
   bSaveMovie = bSaveMovie & ~isempty(fhMonitorFunction);
end

bContinueSim = false;
bSaveState = false;
if (exist('strContinueBase', 'var'))
   bSaveState = true;
   strBaseFilename = [strContinueBase '_'];
   nSplitNumber = 1;

   if (~exist([strContinueBase '_1.mat'], 'file'))
      disp('--- SimulateNetwork: No saved state exists.  Starting from scratch.');
   else
      bContinueSim = true;
   end
end


vsClasses = sNetwork.vsClasses;
nNumClasses = numel(vsClasses);
nNumNeurons = numel(sNetwork.vsNeurons);

vnNeuronClasses = [sNetwork.vsNeurons.nClass];


% - Compute weight matrix

if (~isfield(sNetwork, 'mfWeights') || isempty(sNetwork.mfWeights))
   sNetwork.mfWeights = CalculateWeights(sNetwork);
end



% - Set up initial conditions


if (~bContinueSim)
   ctfActivity = cell(nNumClasses, 1);
   coState = cell(nNumClasses, 1);
   
   for (nClass = 1:nNumClasses)
      fhActFunc = vsClasses(nClass).fhActivationFunction;
      sActParams = vsClasses(nClass).sActivationParams;
      nClassSize = nnz(vnNeuronClasses == nClass);
      
      if (isfield(sNetwork.vsClasses(nClass), 'fhInitialStateFunction') && ...
            ~isempty(sNetwork.vsClasses(nClass).fhInitialStateFunction))
         % - Determine initial state for this class
         vfInitState = sNetwork.vsClasses(nClass).fhInitialStateFunction([nClassSize 1], sNetwork.vsClasses(nClass).cInitialStateParams{:});
      else
         vfInitState = zeros(nClassSize, 1);
      end
      
      % - Compute initial activity and set up class state
      [ctfActivity{nClass}, coState{nClass}] = fhActFunc([], vfInitState, tTimeStep, sActParams);
   end
end

% - Was an input function provided?

if (~exist('fhInputFunction', 'var') || isempty(fhInputFunction))
   fhInputFunction = @(s,t) zeros(numel(s), 1);
end

% - Extract activation and noise functions

for (nClass = 1:nNumClasses) %#ok<FORPF>
   cfhActivationFunction{nClass} = GetParam(vsClasses(nClass), 'fhActivationFunction'); %#ok<AGROW>
   csActivationParams{nClass} = GetParam(vsClasses(nClass), 'sActivationParams'); %#ok<AGROW>
   
   chActNoiseFunction{nClass} = GetParam(vsClasses(nClass), 'fhActNoiseFunc'); %#ok<AGROW>
   cActNoiseParams{nClass} = GetParam(vsClasses(nClass), 'cActNoiseParams'); %#ok<AGROW>
   vfActNoiseWeight(nClass) = GetParam(vsClasses(nClass), 'fActNoiseWeight'); %#ok<AGROW>
end


% - Simulate network

if (bSaveMovie)
   oAVIMovie = avifile(['network_simulation_' datestr(now, 30) '.avi']);
end

if (~isempty(fhMonitorFunction))
   hMonitorFigure = figure;
end

if (bContinueSim)
   % - Read files
   while (exist([strBaseFilename num2str(nSplitNumber) '.mat'], 'file'))
      load([strBaseFilename num2str(nSplitNumber) '.mat'], 'nTimeStep');
      nSplitNumber = nSplitNumber + 1;
   end
   
   % - Load state
   fprintf(1, 'SimulateSheet: Continuing from [%s]\n', [strBaseFilename num2str(nSplitNumber) '.mat']);
   load([strBaseFilename num2str(nSplitNumber-1) '.mat'], ...
      'ctfActivity', 'coState', 'vnSampleNeurons', 'mfResponseTrace', 'mfStateTrace');
   tTime = nTimeStep*tTimeStep; %#ok<NODEF>
   
else
   tTime = 0;
   nTimeStep = 1;
   vfTotalInput = zeros(1, nNumNeurons);
end

nNextSaveStep = nTimeStep + DEF_nSaveStatePeriod;


if (exist('vnSampleNeurons', 'var') && ~isempty(vnSampleNeurons))
   bStoreState = true;
   
   if (~bContinueSim)
       nNumSteps = numel(0:tTimeStep:tSimDuration);
       mfResponseTrace = zeros(numel(vnSampleNeurons), nNumSteps+10);
       mfStateTrace = zeros(numel(vnSampleNeurons), nNumSteps+10);
   end
   
else
   bStoreState = false;
   mfResponseTrace = [];
   mfStateTrace = [];
end


fprintf(1, 'SimulateSheet: %10.2f of %10.2f ms', tTime / 1e-3, tSimDuration / 1e-3);

mfWeights = sNetwork.mfWeights;
bMappedWeights = isa(mfWeights, 'MappedTensor');

while (tTime <= tSimDuration)
   % - Determine recurrent network input
   vfActivity = vertcat(ctfActivity{:});
      
   if (bMappedWeights)
      vfNetworkInput = zeros(size(vfActivity));
      for (nRow = 1:nNumNeurons)
         vfNetworkInput(nRow) = sum(mfWeights(nRow, :)' .* vfActivity);
      end
   else
      vfNetworkInput = mfWeights * vfActivity;
   end
      
   % - Examine partition stability
%    vbActive = vfActivity>0;
%    WP = sNetwork.mfWeights(vbActive, vbActive);
%    J = (WP-eye(size(WP))) ./ 10e-3;
%    vfEigs = eig(J);
%    fMaxEig = max(real(vfEigs));
%    figure(101);
%    cla;
%    plot(vfEigs, 'k.');
%    vfAxis = axis;
%    hold on;
%    plot([0 0], vfAxis(3:4), 'k:');
%    
%    if (fMaxEig > 0)
%       fprintf(1, '\b\b !');
%    end
   
   % - Compute input for this step
   vfExternalInput = fhInputFunction(sNetwork, tTime, tTimeStep);

   % - Iterate over classes
   for (nClass = nNumClasses:-1:1)
      % - Find neurons of this class
      vbThisClass = vnNeuronClasses(:) == nClass;
      nClassSize = nnz(vbThisClass);
      
      % - Add network input
      vfClassInput = vfExternalInput(vbThisClass) + vfNetworkInput(vbThisClass);
      
      % - Add noise
      vfClassInput = vfClassInput + vfActNoiseWeight(nClass) .* chActNoiseFunction{nClass}(cActNoiseParams{nClass}{:}, tTimeStep, [nClassSize 1]);
      
      % - Evaluate class activity
      [ctfActivity{nClass}, coState{nClass}] = cfhActivationFunction{nClass}(coState{nClass}, vfClassInput, tTimeStep, csActivationParams{nClass});
      
      % - Record total input
      vfTotalInput(vbThisClass) = vfClassInput;
      
      ctfFinalAct{nClass} = ctfActivity{nClass};
      ctfFinalState{nClass} = coState{nClass}.fhGetState(coState{nClass});
   end
   
   % - Call monitor function (if requested)
   if (~isempty(fhMonitorFunction))
%       figure(hMonitorFigure);
      fhMonitorFunction(sNetwork, ...
                        ctfActivity, coState, ...
                        vfTotalInput, vfExternalInput, ...
                        tTime, tTimeStep, hMonitorFigure);
   end
   
   % - Save a movie frame (if requested)
   if (bSaveMovie)
      oAVIMovie = addframe(oAVIMovie, gcf);
   end
   
   % - Store state (if requested)
   if (bStoreState)
      vfThisAct = vertcat(ctfFinalAct{:});
      vfThisState = vertcat(ctfFinalState{:});
      mfResponseTrace(:, nTimeStep) = vfThisAct(vnSampleNeurons);
      mfStateTrace(:, nTimeStep) = vfThisState(vnSampleNeurons);
   end
   
   % - Show some feedback
   if (mod(nTimeStep, 1000) == 0)
      fprintf(1, '\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b%10.2f of %10.2f ms', tTime / 1e-3, tSimDuration / 1e-3);
      drawnow;
   end
   
   if (bSaveState && (nTimeStep == nNextSaveStep))
      save([strBaseFilename num2str(nSplitNumber) '.mat'], ...
         'ctfActivity', 'coState', 'vnSampleNeurons', 'mfResponseTrace', 'mfStateTrace', 'nTimeStep');
      nSplitNumber = nSplitNumber + 1;
      nNextSaveStep = nNextSaveStep + DEF_nSaveStatePeriod;
   end
   
   tTime = tTime + tTimeStep;
   nTimeStep = nTimeStep + 1;
end

fprintf(1, '\n');

% - Close the AVI file
if (bSaveMovie)
   oAVIMovie = close(oAVIMovie); %#ok<NASGU>
end

% - Save the final state
if (bSaveState)
   sNetworkOld = sNetwork;
   sNetwork = rmfield(sNetwork, 'mfWeights');
   save([strBaseFilename num2str(nSplitNumber) '.mat'], ...
      'ctfActivity', 'coState', 'vnSampleNeurons', 'mfResponseTrace', 'mfStateTrace', ...
      'vfTotalInput', 'tTimeStep', 'sNetwork', 'tSimDuration', 'strContinueBase');
   sNetwork = sNetworkOld;
end

% --- END of SimulateNetwork.m ---
