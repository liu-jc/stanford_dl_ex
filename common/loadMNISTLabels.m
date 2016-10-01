function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');
%fprintf('size(labels,1) = %d, size(labels,2) = %d\n',size(labels,1),size(labels,2));
%fprintf('numLabels = %d\n',numLabels);
assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end
