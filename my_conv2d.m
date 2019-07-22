function R = my_conv2d(A, B, shape)
% MY_CONV2D is objected to simulate the pre-built conv2 function.
% A = the 2D matrix of image.
% B = the 2D matrix of filter.
% Example:
% A2 = my_conv2d(randi(5,5), randi(3,3), 'full');    
% Copyright 2019 Margorp.

%% arguments checking
    if nargin < 3, shape = 'full'; end % default shape is "full"
    if nargin < 2, error('my_conv2 is required at least two parameters: image matrix and filter matrix.'); end

%% invert fliter B
    B = flip(B,2);
    B = flip(B,1);

%% get or set the dimension of A, B and R, where R is the final matrix in 'full' shape
    [Ah, Aw] = size(A);
    [Bh, Bw] = size(B);
    Rh = Ah + Bh - 1;
    Rw = Aw + Bw - 1;
    
%% add zeros borders to A
    A_ = [zeros(size(A,1), Bw-1), A, zeros(size(A,1), Bw-1)];
    A_ = [zeros(Bh-1, size(A_,2)); A_; zeros(Bh-1, size(A_,2))];
    
%% start calculate the convolution layer result into result matrix
    R = zeros(Rh,Rw);
    for y = 1:Rh
        for x = 1:Rw
            R(y,x) = sum(sum(B .* A_(y:y+Bh-1, x:x+Bw-1)));
        end
    end

%% check if resize the dimension of result matrix is required
    switch shape
        case 'full'
            % do nothing
        case 'same'
            % get an inner matrix in 'same' shape of A
            start_x = ceil((Rw - Aw) / 2) + 1;
            end_x = Aw + start_x - 1;
            start_y = ceil((Rh - Ah) / 2) + 1;
            end_y = Ah + start_y - 1;
            R = R(start_y:end_y, start_x:end_x);
        case 'valid'
            % get an even smaller matrix in 'valid' shape
            start_x = (Rw-Aw) + 1;
            end_x = start_x + Rw - (Rw-Aw)*2 - 1;
            start_y = (Rh-Ah) + 1;
            end_y = start_y + Rh - (Rh-Ah)*2 - 1;
            R = R(start_y:end_y, start_x:end_x);
        otherwise
            error('Invalid shape: %s\nAcceptable return shapes are "full", "same", "valid".', shape);
    end
    
end
