
%%
[nameStart, filepath] = uigetfile('.tif', "Choose the first image in the sequence.");

%%

nameStart = 'map1_000001.tif';
filepath = 'D:\projects\neurotar_eye_tracking\widefield_mapping\230104_DMM001_SignMapping\';

%%
splitName = split(nameStart, '_');
keepNameParts = splitName(1:end-1,1);
baseName = string(join(keepNameParts));

imgList = dir(join([filepath, '\', baseName, '*.tif'], ''));

%%

block_size = 1000;
num_blocks = ceil(size(imgList,1) / block_size);

startF = 1;
for b = 1:num_blocks

    endF = startF + (block_size-1);
    
    if endF > size(imgList,1)
        endF = size(imgList,1);
    end

    savePath = fullfile(filepath, join([baseName,'_stack_block',b,'.tif'], ''));
    writeTifBlock(imgList(startF:endF), savePath)

    fprintf('Writing block %d of %d (frames %d through %d) \n',b,num_blocks,startF,endF)
    
    startF = endF+1; 

end

%%

