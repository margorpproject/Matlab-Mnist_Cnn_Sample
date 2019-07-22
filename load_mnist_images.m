function [images,num_images,num_rows,num_cols] = load_mnist_images(filename)
    fp = fopen(filename, 'rb');
    assert(fp ~= -1, 'Failed to open file at %s', filename);
    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2051, 'Invalid magic number in %s with magic number as %d', filename, magic);
    num_images = fread(fp, 1, 'int32', 0, 'ieee-be');
    num_rows = fread(fp, 1, 'int32', 0, 'ieee-be');
    num_cols = fread(fp, 1, 'int32', 0, 'ieee-be');
    images = fread(fp, inf, 'unsigned char=>unsigned char');
    images = reshape(images, num_rows, num_cols, num_images);
    images = permute(images, [2 1 3]);
    images = reshape(images, numel(images), 1);
    images = double(images) / 255;
    fclose(fp);
end