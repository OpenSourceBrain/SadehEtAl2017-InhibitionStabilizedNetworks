function [fStimDeriv, fNonStimDeriv] = NumericPerturbationDeriv(nNumStim, nNumInh, sNetwork, bFPSolver)

fTOL = 1e-2;
tMaxTime = 6400e-3;

if (~exist('bFPSolver', 'var'))
   bFPSolver = true;
end

fBoostDelta = 0.01;

nNumNeurons = size(sNetwork.mfWeights, 1);

nNumStim = round(nNumStim);
if (nNumStim > nNumInh) || (nNumStim < 1)
   fStimDeriv = nan;
   return;
end

vfInputBase = ones(nNumNeurons, 1);
vfInputBoost = fBoostDelta .* [zeros(nNumNeurons-nNumStim, 1); ones(nNumStim, 1)];

if (bFPSolver)
   fhFPResp = @(t, y, vfI)sum((sNetwork.mfWeights * y - y + vfI))./nNumNeurons;
   
   options = optimoptions('fminunc', 'Display', 'none', 'Algorithm', 'Quasi-Newton');
   
   vfBaseResp = fminunc(@(x)fhFPResp(0, x, vfInputBase), vfInputBase .* nNumNeurons + rand(nNumNeurons, 1), options);
   vfBoostResp = fminunc(@(x)fhFPResp(0, x, vfInputBase + vfInputBoost), vfBaseResp - vfInputBoost .* rand(nNumNeurons, 1), options);
   
else  % - Use Euler solver
   tTimeStep = 1e-3;
   
   vfCurrentState = vfInputBase;
   tCurrentDur = 100e-3;
   
   while true
      [~, ~, ~, ~, ~, mfThisStateTrace] = ...
         SimulateNetwork(sNetwork, [], tTimeStep, tCurrentDur, @(~,~,~)vfInputBase, false, [1:nNumNeurons], vfCurrentState);
      vfCurrentState = mfThisStateTrace(:, end);
      vfBaseResp = mfThisStateTrace(:, end);
      
      % - Test for fixed point
      if sum(abs(diff(mfThisStateTrace(:, end-1:end), [], 2))) < fTOL
         % - We can finish here
         break;
      else
         % - Continue from current point, with longer duration
         tCurrentDur = tCurrentDur * 2;
         
         if (tCurrentDur > tMaxTime)
            % - Give up
            vfBaseResp = nan(size(vfBaseResp));
            break;
         end
         
         continue;
      end
   end
   
   while true
      [~, ~, ~, ~, ~, mfThisStateTrace] = ...
         SimulateNetwork(sNetwork, [], tTimeStep, tCurrentDur, @(~,~,~)vfInputBase + vfInputBoost, false, [1:nNumNeurons], vfCurrentState);
      vfCurrentState = mfThisStateTrace(:, end);
      vfBoostResp = mfThisStateTrace(:, end);
      
      % - Test for fixed point
      if sum(abs(diff(mfThisStateTrace(:, end-1:end), [], 2))) < fTOL
         % - We can finish here
         break;
      else
         % - Continue from current point, with longer duration
         tCurrentDur = tCurrentDur * 2;
         
         if (tCurrentDur > tMaxTime)
            % - Give up
            vfBaseResp = nan(size(vfBaseResp));
            break;
         end
         
         continue;
      end
   end
end

vfStimDiff = vfBoostResp - vfBaseResp;

fStimDeriv = nanmean(vfStimDiff(end-nNumStim+1:end)) ./ fBoostDelta;
fNonStimDeriv = nanmean(vfStimDiff(1:end-nNumInh)) ./ fBoostDelta;

% figure(101);
% hold on;
% plot(nNumStim/nNumInh*100, fStimDeriv, 'kx');

return;

%%

tTimeStep = 1e-3;
tStimDelay = 100e-3;

fhInputFunction = @(sN, tT, tDelta)vfInputBase + vfInputBoost .* (tT > tStimDelay);
fhMonitorFunction = @(varargin)MonitorTrace(varargin{:}, [1 nNumNeurons]);

[~, ~, ~, ~, ~, mfStateTrace] = ...
   SimulateNetwork(sNetwork, [], tTimeStep, tStimDelay*2, fhInputFunction, false, [1 nNumNeurons]);
nStimSteps = size(mfStateTrace, 2);
vfBaseResp = mfStateTrace(:, floor(nStimSteps/2));
vfBoostResp = mfStateTrace(:, nStimSteps);
vfStimDiff = vfBoostResp - vfBaseResp;

bBelowThresh = any(vfBoostResp <= 0);
