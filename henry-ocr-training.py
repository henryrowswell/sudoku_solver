import sys
import os
import numpy as np
import cv2

#want to be able to feed it a lot of sudoku puzzles for training
#then need a function I can use to pass in an image / numpy array, and get the number out
#then use that function to build the sudoku puzzle
#then use my algorithm to solve the puzzle
indir = "D:/Documents/sudokupy/henry-ocr/training_imgs"
# cv2.namedWindow('hello',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('hello', 600,600)

for root, dirs, filenames in os.walk(indir):
    for f in filenames:
        print("current file: " + repr(f))
        im = cv2.imread(root + "/" + f)
        # im3 = im.copy()

        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2) #why doesn't this work for finding individual digits when not inverted?
        thresh2 = cv2.adaptiveThreshold(blur,255,1,1,11,2) #cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

        # cv2.imshow('thresh',thresh)
        # cv2.imshow('thresh2',thresh2)

        #################      Now finding Contours         ###################

        image,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #used to be RETR_LIST (what is that?)
        cv2.drawContours(im, contours, -1, (0,255,0), 1)
        #cv2.imshow('hello2',thresh)
        #cv2.waitKey(0)

        samples =  np.empty((0,100))
        responses = []
        keys = [i for i in range(48,58)]

        for cnt in contours:
            #print(cv2.contourArea(cnt))
            [x,y,w,h] = cv2.boundingRect(cnt)
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2) #just draws rectangle for user
            roi = thresh2[y:y+h,x:x+w] #wtf this was using thresh which is very faint after adaptive thresholding WHY IS THIS BETTER WITH THRESH THAN THRESH2
            roismall = cv2.resize(roi,(10,10))
            #cv2.imshow("current_roismall", roismall)
            cv2.imshow("current_roi", roi)
            cv2.imshow('input',im)
            #cv2.imshow("thresh2", thresh2)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))


with open('generalsamples.data','a') as f_generalsamples, open('generalresponses.data','a') as f_generalresponses:
    np.savetxt(f_generalsamples,samples)
    np.savetxt(f_generalresponses,responses)

print "training complete"

#######   training part    ###############
# samples = np.loadtxt('generalsamples.data',np.float32)
# responses = np.loadtxt('generalresponses.data',np.float32)
# responses = responses.reshape((responses.size,1))
#
# model = cv2.KNearest()
# model.train(samples,responses)
