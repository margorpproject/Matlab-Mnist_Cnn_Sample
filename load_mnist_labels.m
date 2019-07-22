function labels = load_mnist_labels(filename)
    fid = fopen(filename, 'rb');
    assert(fid ~= -1, 'Could not open file at %s', filename);
    magic = fread(fid, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2049, 'Invalid magic number %d in file at %s', magic, filename);
    num_labels = fread(fid, 1, 'int32', 0, 'ieee-be');
    labels = fread(fid, inf, 'unsigned char');
    assert(size(labels, 1) == num_labels, 'Mis-match number of labels %d from file %s', num_labels, filename);
    fclose(fid);
end