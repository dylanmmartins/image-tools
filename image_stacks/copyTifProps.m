function [bt, t] = copyTifProps(bt, t)

    % Set up properties to match individual frames
    setTag(bt, "ImageLength", getTag(t,"ImageLength"));
    setTag(bt, "ImageWidth", getTag(t,"ImageWidth"));
    setTag(bt, "Photometric", getTag(t,"Photometric"));
    setTag(bt, "SubFileType", getTag(t,"SubFileType"));
    setTag(bt, "RowsPerStrip", getTag(t,"RowsPerStrip"));
    setTag(bt, "BitsPerSample", getTag(t,"BitsPerSample"));
    setTag(bt, "Compression", getTag(t,"Compression"));
    setTag(bt, "SampleFormat", getTag(t,"SampleFormat"));
    setTag(bt, "SamplesPerPixel", getTag(t,"SamplesPerPixel"));
    setTag(bt, "PlanarConfiguration", getTag(t,"PlanarConfiguration"));
    setTag(bt, "Orientation", getTag(t,"Orientation"));

end