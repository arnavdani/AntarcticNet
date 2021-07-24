import sys
import numpy as np
import cv2
import os



dir_path = 'C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/train_data'


save_path = 'C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/contours'



img_data = open("C:/Users/arnav/OneDrive/Documents/College/Summer 2021/Clarks/MakingEllipse/contourdata.txt", "w+")
    
for filename in os.listdir(dir_path):
    
    
    
    img_path = os.path.join(dir_path, filename)
    
    
    
    image = cv2.imread(img_path)


    # convert the image to grayscale format
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    #cv2.imshow("Original B/W image", img_gray)
    #cv2.waitKey()   
    

    # apply binary thresholding
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    #ret2, thres2 = cv2.threshold(img_gray, 70, 255, cv2.THRESH_BINARY)
    

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    #contours2, hierarchy2 = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method = cv2.CHAIN_APPROX_NONE)
    
                                        
    # draw contours on the original image
    image_copy = image.copy()
    img_copy2 = image.copy()
    
    
    #cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                
    contours = sorted(contours, key=cv2.contourArea, reverse=False)
    #contours2 = sorted(contours2, key = cv2.contourArea, reverse=False)
    
    #cv2.drawContours(img_copy2, contours, -1, (0,255,0), 3)
    
    #cv2.imshow("Image with background removed", img_copy2)
    #cv2.waitKey(0)
    
    rawName = filename.split('.')[0]
    
    save_dir = os.path.join(save_path, rawName)
    
    os.mkdir(save_dir)
    
    
    
    counter = 0

    for i in contours:
        
        
        #print(str(cv2.contourArea(i)))
        
        if (cv2.contourArea(i) > 600 and cv2.contourArea(i) < 50000):
            
            mask = np.zeros(image_copy.shape[:2], np.uint8)
            
            cv2.drawContours(mask, [i],-1, 255, -1)
            
           # cv2.imshow("Image with background removed", mask)
            #cv2.waitKey(0)

            new_img = cv2.bitwise_and(image_copy, image_copy, mask=mask)
            
            
            
            
            newName = rawName + "-contour_" + str(counter)
            newAdr = newName + ".png"
            counter = counter + 1
        
            saveAdr = os.path.join(save_dir , newAdr)
            
            img_data.write(str(newAdr) + " " + str(int(cv2.contourArea(i))) + " \n")
            
            #print("saving " + str(newAdr))
            cv2.imwrite(saveAdr, new_img)
            #print(saved)

            


    

        
