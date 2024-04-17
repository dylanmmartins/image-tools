

import numpy as np
import cv2

def avi_to_arr(path, ds=1.0):

    vid = cv2.VideoCapture(path)
    
    # array to put video frames into
    # will have the shape: [frames, height, width] and be
    # returned with dtype=int8
    arr = np.empty(
        [int(vid.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)*ds),
        int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)*ds)],
        dtype=np.uint8)
    
    # iterate through each frame
    for f in range(0,int(vid.get(cv2.CAP_PROP_FRAME_COUNT))):
        
        # read the frame in and make sure it is read in correctly
        ret, img = vid.read()
        if not ret:
            break
        
        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # downsample the frame
        img_s = cv2.resize(
            img,
            (0,0),
            fx=ds,
            fy=ds,
            interpolation=cv2.INTER_NEAREST
        )
        
        # add the downsampled frame to all_frames as int8
        arr[f,:,:] = img_s.astype(np.int8)

    return arr