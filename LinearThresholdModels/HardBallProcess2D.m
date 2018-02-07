function [mfCentres] = HardBallProcess2D(nNumBalls, vfSpace, fBallDiameter)

% HardBallProcess2D - FUNCTION Place hard balls in a 2D space
%
% Usage: [mfCentres] = HardBallProcess2D(nNumBalls, vfSpace, fBallDiameter)
%
% This function simulates a Poisson-like "hard ball" process -- that is, a
% spatial random process with a hard limit on how close two elements can be.  If
% 'fBallDiameter' is zero, this is a 2D Poisson process.  Note that for silly
% combinations of parameters, this function will not be able to find a solution
% and will not halt...

% Author: Dylan Muir <dylan@ini.phys.ethz.ch>
% Created: 15th September, 2010

% -- Defaults
DEF_fBallDiameter = 0;


% -- Check arguments

if (~exist('fBallDiameter', 'var'))
   fBallDiameter = DEF_fBallDiameter;
end 


% -- Draw random locations

% - Initialise centres
mfCentres = inf(0, 2);

nNumPlaced = 0;
while (nNumPlaced < nNumBalls)
   % - Generate some random locations
   mfTheseCentres = rand(nNumBalls - nNumPlaced, 2) .* repmat(vfSpace, nNumBalls - nNumPlaced, 1);

   % Vet random locations by inter-location distance
   if (fBallDiameter > 0)
      mfDistances = squareform(pdist([mfCentres; mfTheseCentres]));
      mfDistances(mfDistances == 0) = inf;
      vbReject = any(mfDistances < fBallDiameter);
   else
      vbReject = false(size(mfTheseCentres, 1), 1);
   end

   % - Construct locations list
   mfCentres = [mfCentres; mfTheseCentres(~vbReject(nNumPlaced+1:end), :)]; %#ok<AGROW>
   nNumPlaced = size(mfCentres, 1);
end

% --- END of HardBallProcess2D.m ---
