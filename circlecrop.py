import os
import sys 
from findellipse import find_ellipse

path_to_module = 'C:/Users/arnav/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0/LocalCache/local-packages/Python39/site-packages'

sys.path.append(path_to_module)

import numpy as np
import cv2


dir_path = 'C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/input_data'
dir_save = './output_data'


for folder in os.listdir(dir_path):
    print(folder)
    img_path = os.path.join(dir_path , folder)    
    for filename in os.listdir(img_path):
        
        
        image = cv2.imread(os.path.join(img_path , filename))
        #print(filename)
        
        #cv2.imshow('image', image)

        width = 1000
        length = 1000
        dim = (width, length)
        image = cv2.resize(image, dim)

        # convert the image to grayscale format
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        #cv2.imshow("Original B/W image", img_gray)
        #cv2.waitKey()   

        # apply binary thresholding
        ret, thresh = cv2.threshold(img_gray, 155, 255, cv2.THRESH_BINARY)
        #PARAMETER - 150
        # visualize the binary image
    
        #cv2.imshow('Binary image', thresh)
        #cv2.waitKey(0)
        #cv2.imwrite('image_thres1.jpg', thresh)
        #cv2.destroyAllWindows()


        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, heirarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                        
        # draw contours on the original image
        image_copy = image.copy()
        #cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        ell = None
        for i in contours:
            ell = find_ellipse(i, image_copy)
            if not(ell is None):
                #if cv2.contourArea(C) > 30000 and cv2.arcLength(C, True) < 6000:
                #    break
                dims = ell[1]  # bounding rectangle dimensions
                w = dims[0]
                h = dims[1]
                if w*h > 0.5*width*length:
                    break
                #PARAMETER - 0.5

        mask = np.zeros(image_copy.shape[:2], np.uint8)

        if not(ell is None):
            cv2.ellipse(mask, ell, 255, -1)
        
            #cv2.imshow('Binary image', mask)
            #cv2.waitKey(0)
            
            
            #cv2.imwrite('image_thres1.jpg', thresh)
            
            #cv2.destroyAllWindows()

            new_img = cv2.bitwise_and(image_copy, image_copy, mask=mask)
            #cv2.waitKey(0)

            #cv2.imshow('Binary image', new_img)
            #cv2.waitKey(0)
        
            #cv2.imshow("Image with background removed", new_img)
            cv2.imwrite(os.path.join(dir_save , filename), new_img)
            print(os.path.join(dir_save , filename))
            #cv2.waitKey(0)






