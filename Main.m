tic;

test_images_filepath = 'I:/Storage/Work/Project/Language/Matlab/Data/t10k-images.idx3-ubyte';
test_labels_filepath = 'I:/Storage/Work/Project/Language/Matlab/Data/t10k-labels.idx1-ubyte';
train_images_filepath = 'I:/Storage/Work/Project/Language/Matlab/Data/train-images.idx3-ubyte';
train_labels_filepath = 'I:/Storage/Work/Project/Language/Matlab/Data/train-labels.idx1-ubyte';

[train_images,num_train_images,num_train_rows,num_train_cols] = load_mnist_images(train_images_filepath);
train_labels = load_mnist_labels(train_labels_filepath);
[test_images,num_test_images,num_test_rows,num_test_cols] = load_mnist_images(test_images_filepath);
test_labels = load_mnist_labels(test_labels_filepath);

% fprintf('num_train_images: %d | num_train_rows: %d | num_train_cols: %d\n', ...
% num_train_images, num_train_rows, num_train_cols);

% fprintf('num_test_images: %d | num_test_rows: %d | num_test_cols: %d\n', ...
% num_test_images, num_test_rows, num_test_cols);
    
% return

train_images = reshape(train_images, num_train_rows, num_train_cols, num_train_images);
train_labels(train_labels==0) = 10;

% disp(size(train_images))
% disp(size(train_labels))

test_images = reshape(test_images, num_test_rows, num_test_cols, num_test_images);
test_labels(test_labels==0) = 10;

% disp(size(test_images))
% disp(size(test_labels))

% return

num_train = floor(num_train_images * 0.8);
num_test = num_train_images - num_train;
% fprintf('num_train: %d | num_test: %d\n', num_train, num_test);

rand_indexes = randperm(num_train_images);
rand_train_indexes = rand_indexes(1:num_train);
rand_test_indexes = rand_indexes(num_train+1:end);

% return

rng(1);

W1 = 1e-2 * randn([9, 9, 20]);
W5 = (2*rand(100, 2000) - 1) * sqrt(6) / sqrt(360 + 2000);
Wo = (2*rand(10, 100) - 1) * sqrt(6) / sqrt(10 * 100);

X = train_images(:,:,rand_train_indexes);
D = train_labels(rand_train_indexes);

for epoch = 1:3
    epoch
    [W1, W5, Wo] = Mnist_Conv(W1, W5, Wo, X, D);
end

save('Mnist_Conv.mat');

X = train_images(:, :, rand_test_indexes);
D = train_labels(rand_test_indexes);
acc = 0;
N = length(D);

for k = 1:N
    x = X(:, :, k);
    y1 = Conv(x, W1);
    y2 = ReLU(y1);
    y3 = Pool(y2);
    y4 = reshape(y3, [], 1);
    v5 = W5 * y4;
    y5 = ReLU(v5);
    v = Wo * y5;
    y = Softmax(v);
    
    [~, i] = max(y);
    if i == D(k)
        acc = acc + 1;
    end
end

acc = 100 * acc / N;
fprintf('Accuracy is %f%%\n', acc);

toc;