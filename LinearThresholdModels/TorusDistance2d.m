function [vfDist, mfAllDistances] = TorusDistance2d(vMeshSize, mSourcePoints, mDestPoints)

% TorusDistance2d - FUNCTION Calculate the euclidean distance on a quasi-torus
%
% Usage: [vfDist, mfAllDistances] = TorusDistance2d(vMeshSize, mSourcePoints, mDestPoints)
%
% Where: 'vMeshSize' is a vector with two elements, specifying the X and Y
% dimensions of the mesh, respectively.  'mSourcePoints' is a matrix, where
% each row is a two-element vector specifying a point in the space defined
% by 'vMeshSize'.  'mDestPoints' is a matrix of points with the same format
% as 'mSourcePoints'.  'mSourcePoints' and 'mDestPoints' must be the same
% size.
%
% TorusDistance2d will calculate the shortest distance between each pair of
% points, taken row-wise from 'mSourcePoints' and 'mDestPoints',
% considering that the mesh forms a quasi-torus such that the left and
% right edges are joined, and the top and bottom edges are joined.  This is
% not exactly the distance on the surface of a torus, as mesh elements in
% the middle of the mesh are spaced equally with elements at the edges of
% the mesh.
%
% 'vfDist' will be a vector with the same number of elements as
% 'mSourcePoints' has coordinate pairs.  Each element in 'vfDist' will be
% the shortest euclidean distance between the corresponding pair of points
% from 'mSourcePoints' and 'mDestPoints', computed on the quasi-torus
% surface.

% Author: Dylan Muir <dylan@ini.phys.ethz.ch> (with thanks to Chiara
%           Bartalozzi)
% Created: 31st May, 2006
% Copyright 2006 Dylan Muir

% -- Check arguments

if (nargin > 3)
   disp('--- TorusDistance2d: Extra arguments ignored');
end

if (nargin < 3)
   disp('*** TorusDistance2d: Incorrect usage');
   help TorusDistance2d;
   return;
end

% -- Check size of arguments

% - Is vMeshSize a two-element vector specifying the size of the mesh?
if (numel(vMeshSize) ~= 2)
   disp('*** TorusDistance2d: ''vMeshSize'' must be a two-element vector');
   return;
end

% - Are mSourcePoints and mDestPoints the same size or scalar coords?
if (~all(size(mSourcePoints) == size(mDestPoints)) && (size(mDestPoints, 1) ~= 1) && (size(mSourcePoints, 1) ~= 1))
   disp('*** TorusDistance2d: ''mSourcePoints'' and ''mDestPoints'' must be the same size');
   return;
end

% - The coordinate arrays should be... well, coordinates.  Therefore at
% least one dimension length should be two
if (~any(size(mSourcePoints) == 2))
   disp('*** TorusDistance2d: Coordinate arrays ''mSourcePoints'' and ''mDestPoints''');
   disp('       must contain pairs of coordinates');
   return;
end

% - Transpose the coordinate arrays, if they're in the wrong orientation
if (size(mSourcePoints, 2) > 2)
   mSourcePoints = mSourcePoints';
   mDestPoints = mDestPoints';
end

% - Extract number of point pairs
nNumPoints = max(size(mSourcePoints, 1), size(mDestPoints, 1));

% - Are all points within the mesh?  (Assume so for speed)
% if (any(any(mSourcePoints < 0)) || any(any(mDestPoints < 0)) || ...
%       any(any(mSourcePoints > repmat(vMeshSize, nNumPoints, 1))) || ...
%       any(any(mDestPoints > repmat(vMeshSize, nNumPoints, 1))))
%    disp('*** TorusDistance2d: All points must fall within the space defined by');
%    disp('       (0, 0) and ''vMeshSize''');
%    return;
% end


% -- Calculate manhattan block edge distances for pairwise points
mWithinMeshDist(:, 2) = abs(mSourcePoints(:, 2) - mDestPoints(:, 2));
mWithinMeshDist(:, 1) = abs(mSourcePoints(:, 1) - mDestPoints(:, 1));

% - The distance of the source point to the origin is just the coordinate
% of the point itself.  So to calculate the distance wrapped over the edge
% of the mesh, add the distance between the destination and the mesh edge
mWrapUpDist(:, 2) = mDestPoints(:, 2) + abs(vMeshSize(2) - mSourcePoints(:, 2));
mWrapUpDist(:, 1) = mDestPoints(:, 1) + abs(vMeshSize(1) - mSourcePoints(:, 1));
mWrapDownDist(:, 2) = mSourcePoints(:, 2) + abs(vMeshSize(2) - mDestPoints(:, 2));
mWrapDownDist(:, 1) = mSourcePoints(:, 1) + abs(vMeshSize(1) - mDestPoints(:, 1));
mMinWrapDist = min(mWrapUpDist, mWrapDownDist);

% - Find minimum distances for x and y
mManhattanDist(:, 1) = min([mWithinMeshDist(:, 1) mMinWrapDist(:, 1)], [], 2);
mManhattanDist(:, 2) = min([mWithinMeshDist(:, 2) mMinWrapDist(:, 2)], [], 2);

% - Compute euclidean distance
vfDist = sqrt(sum(mManhattanDist.^2, 2));

% - Find rejected secondary distance over the torus (either wrapped or within the mesh)
if (nargout > 1)
   % - "Within the mesh" distance
   mfAllDistances(:, 1) = sqrt(sum(mWithinMeshDist.^2, 2));
   
   % - "Wrap and within" distance
   mfAllDistances(:, 2) = sqrt(mMinWrapDist(:, 1).^2 + mWithinMeshDist(:, 2).^2);

   % - "Within and wrap" distance
   mfAllDistances(:, 3) = sqrt(mWithinMeshDist(:, 1).^2 + mMinWrapDist(:, 2).^2);
   
   % - "Over the border" distance
   mfAllDistances(:, 4) = sqrt(mMinWrapDist(:, 1).^2 + mMinWrapDist(:, 2).^2);
end

% --- END TorusDistance2d.m ---
