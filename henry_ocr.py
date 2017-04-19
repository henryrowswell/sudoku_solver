import sys
import numpy as np
import cv2

#######   training part    ###############
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.KNearest()
model.train(samples,responses)

def get_number_from_img(im, draw=False):
    #im = cv2.imread(im)
    #out = np.zeros(im.shape,np.uint8)
    #gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
    #thresh2 = cv2.adaptiveThreshold(gray,255,1,1,11,2)
    thresh = im
    thresh2 = np.copy(im)
    # cv2.imshow('OCR',thresh2)
    # cv2.waitKey(0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    #get biggest contour (the number)
    biggest = None
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > max_area:
            biggest = i
            max_area = area

    if draw:
        #cv2.drawContours(im, contours, -1, (0,255,0), 1)
        cv2.drawContours(im, [biggest], 0, (0,255,0), 5)
        cv2.imshow('im',im)
        cv2.waitKey(0)

    output = ""
    if biggest != None:
        cnt = biggest
        [x,y,w,h] = cv2.boundingRect(cnt)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        roi = thresh2[y:y+h,x:x+w]
        roismall = cv2.resize(roi,(10,10))
        roismall = roismall.reshape((1,100))
        roismall = np.float32(roismall)
        retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
        string = str(int((results[0][0])))
        #cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
        output += string

    return output
