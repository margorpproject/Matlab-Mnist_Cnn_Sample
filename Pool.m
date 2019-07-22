function y = Pool(x)
    [x_row, x_col, num_filters] = size(x);
    y = zeros(x_row/2, x_col/2, num_filters);
    for k = 1:num_filters
        filter = ones(2) / (2*2);
        image = conv2(x(:,:,k), filter, 'valid');
        y(:,:,k) = image(1:2:end, 1:2:end);
    end
end