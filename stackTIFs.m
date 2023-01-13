function stackTIFs()
%-------------------------------------------------------------------------%
%
%   Take a list of individual TIF images and write them out as a single TIF
%   stack. The images will be kept in stacks of 1000 images in order to
%   keep file sizes reasonable. This block size can be changed.
%
%   When you run this function, a dialog box will open. Pick the first
%   image in the sequence, which should be in a folder with the rest of the
%   images.
%
%   Written Jan 05 2023 by DMM
%
%-------------------------------------------------------------------------%


    [nameStart, filepath] = uigetfile('.tif', "Choose the first image in the sequence.");

    splitName = split(nameStart, '_');
    keepNameParts = splitName(1:end-1,1);
    baseName = string(join(keepNameParts));
    
    imgList = dir(join([filepath, '\', baseName, '*.tif'], ''));

    stimFile = uigetfile('.mat', "Choose the stimulus file.");
    load(stimFile, "Stimdata");
    
    num_blocks = Stimdata.repeats;
    block_size = floor(size(imgList,1) / 10);

    % block_size = 1000;
    % num_blocks = ceil(size(imgList,1) / block_size);
    
    startF = 1;
    for b = 1:num_blocks
    
        endF = startF + (block_size-1);

        if endF > size(imgList,1)
            endF = size(imgList,1);
        end

        fprintf('Writing block %d of %d (frames %d through %d) \n',b,num_blocks,startF,endF)
    
        savePath = fullfile(filepath, join([baseName,'_stack_block',b,'.tif'], ''));
        writeTifBlock(imgList(startF:endF), savePath);
        
        startF = endF+1;

        Stimdata.repeats = 1;

        save(string(join([filepath, 'split_stim_file.mat'])), Stimdata)
    
    end

end