    import sys
    import numpy as np
    import cv2
    import os



    file_path = 'Desktop/College/Summer 2021/Clarks/MakingCircle/circlegen/data/blogplot.png'
    save_path = 'Desktop/College/Summer 2021/Clarks/MakingCircle/circlegen/cropped circles'
    scale = 0.8
    image = cv2.imread(file_path)


    width = int(image.shape[1]*scale)
    length = int(image.shape[0]*scale)
    dim = (width, length)
    image = cv2.resize(image, dim)

    # convert the image to grayscale format
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    cv2.imshow("Original B/W image", img_gray)
    cv2.waitKey()   

    # apply binary thresholding
    ret, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    # visualize the binary image
    cv2.imshow('Binary image', thresh)
    cv2.waitKey(0)
    #cv2.imwrite('image_thres1.jpg', thresh)
    cv2.destroyAllWindows()


    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                        
    # draw contours on the original image
    image_copy = image.copy()
    #cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                
    contours = sorted(contours, key=cv2.contourArea)

    for i in contours:
        
        if cv2.arcLength(i, True) > 400:
            
            break

    mask = np.zeros(image_copy.shape[:2], np.uint8)

    cv2.drawContours(mask, [i],-1, 255, -1)

    new_img = cv2.bitwise_and(image_copy, image_copy, mask=mask)

    cv2.imshow("Image with background removed", new_img)
    cv2.imwrite(os.path.join(save_path , 'plotcropped.png'), new_img)
    cv2.waitKey(0)





