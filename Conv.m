function y = Conv(x, W)
    [w_row, w_col, num_filters] = size(W);
    [x_row, x_col, ~] = size(x);
    y_row = x_row - w_row + 1;
    y_col = x_col - w_col + 1;
    y = zeros(y_row, y_col, num_filters);
    
    for k = 1:num_filters
        filter = W(:,:,k);
        filter = rot90(squeeze(filter), 2);
        y(:,:,k) = conv2(x, filter, 'valid');
    end
end