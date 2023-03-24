function writeTifBlock(imgList, savePath)

    % Read the first image in
    tmpImgPath = fullfile(imgList(1).folder, imgList(1).name);
    t = Tiff(tmpImgPath, 'r');
    im = read(t);
    
    % Write big tif... 'w8' is a larger format that can go past 4GB
    bt = Tiff(savePath,'w8');
    setTag(bt, "ImageLength", getTag(t,"ImageLength"));
    setTag(bt, "ImageWidth", getTag(t,"ImageWidth"));
    setTag(bt, "Photometric", getTag(t,"Photometric"));
    setTag(bt, "PlanarConfiguration", getTag(t,"PlanarConfiguration"));
    setTag(bt, "BitsPerSample", getTag(t,"BitsPerSample"));

    write(bt, im);
    
    close(t);
    
    % the rest of the images will be written into the same file as a stack
    for fnum = 2:(size(imgList,1)-1)
    
        tmpImgPath = fullfile(imgList(fnum).folder, imgList(fnum).name);
        t = Tiff(tmpImgPath, 'r');
        im = read(t);
    
        % add the Tif directory for the new image inside of the file
        writeDirectory(bt);
    
        setTag(bt, "ImageLength", getTag(t,"ImageLength"));
        setTag(bt, "ImageWidth", getTag(t,"ImageWidth"));
        setTag(bt, "Photometric", getTag(t,"Photometric"));
        setTag(bt, "PlanarConfiguration", getTag(t,"PlanarConfiguration"));
        setTag(bt, "BitsPerSample", getTag(t,"BitsPerSample"));
    
        write(bt, im);
    
        close(t);
    
    end
    
    close(bt);

end