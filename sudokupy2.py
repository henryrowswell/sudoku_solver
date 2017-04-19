from PIL import Image
#from pytesser import *
import henry_ocr
import sudoku_solver
#import pytesseract

import cv2
import numpy as np
import time
import math
import operator
import sys

def peek(window, img):
    cv2.imshow(window,img)

def waitToDestroy():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run(input_img):
    # open the image
    #img = cv2.imread(filename)
    #peek("1",img)

    # resize all input images to the same size for convenience
    img = cv2.resize(input_img,(900,900))

    img2 = np.copy(img)

    # convert img to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #peek("2",gray)

    # blur the image to smooth out noise, even when they look fine they have noise sometimes
    smooth = cv2.GaussianBlur(gray,(3,3),0)
    #peek("3",smooth)

    # adaptive threshold
    thresh = cv2.adaptiveThreshold(smooth,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,2)
    thresh2 = cv2.adaptiveThreshold(smooth,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,2)
    #peek("4",thresh)

    # find contour lines - after this somehow the thresh img is changed, which is why we have thresh2
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(img, contours, -1, (0,255,0), 1)
    #peek("3",img)

    # idx = 0 # The index of the contour that surrounds your object
    # mask = np.zeros_like(gray) # Create mask where white is what we want, black otherwise
    # cv2.drawContours(mask, contours, idx, 255, -1) # Draw filled contour in mask
    # out = np.zeros_like(gray) # Extract out the object and place into output image
    # # out[mask == 255] = img[mask == 255]
    #
    # # Show the output image
    # cv2.imshow('Output', out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    selected_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 5000 and cv2.contourArea(cnt) < 15000: #trying to isolate actual grids, should work if all images are same size
            x,y,w,h = cv2.boundingRect(cnt)
            selected_contours.append((x,y,w,h))

    #count if there's 81 here
    if len(selected_contours) != 81:
        print("ERROR: wrong number of contours somehow")
        sys.exit()


    #sort contours by x,y, (starting in top left)
    selected_contours = sorted(selected_contours, key=operator.itemgetter(1,0))
    contour_values = []
    puzzle = [[],[],[],[],[],[],[],[],[]]
    row_count = 0
    #cut out and save
    idx =0
    for x,y,w,h in selected_contours:
        #if cv2.contourArea(cnt) > 10000 and cv2.contourArea(cnt) < 20000:#trying to isolate actual grids HOPEFULLY EXACTLY 81
        idx += 1
        #x,y,w,h = cv2.boundingRect(cnt)
        roi=thresh2[y+5:y+h-5,x+5:x+w-5] #used to be img (which is not pure black/white) #also cropping black edges off
        # cv2.imshow('ROI',roi)
        # cv2.waitKey(0)
        # cv2.imwrite("grid_pics/" + str(idx) + '.jpg', roi) #SAVING PICS FOR ANALYSIS
        #print(str(idx) + ":" + image_file_to_string("grid_pics/" + str(idx) + '.jpg'))

        #cv2.rectangle(img,(x,y),(x+w-2,y+h-2),(0,255,0),1)

        #get the number using my FANCY OCR
        #num = henry_ocr.get_number_from_img("grid_pics/" + str(idx) + '.jpg')
        num = henry_ocr.get_number_from_img(roi)
        #print(str(idx) + ":" + num, x, y)#, cv2.contourArea(cnt))

        #save number or blank into next slot of board data structure
        if num:
            puzzle[row_count / 9].append(num)
            cv2.putText(img2,num,(x+w/8,y+h/3),0,1,(255,0,0),2)
        else:
            puzzle[row_count / 9].append("x")
        row_count += 1

        contour_values.append((x,y,w,h,num)) #this is just used for writing solution to img

    peek("4", img)

    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in puzzle]))

    # cv2.imshow('img',img)
    # cv2.waitKey(0)


    #################TIME TO SOLVE###########################

    #["..9748...","7........",".2.1.9...","..7...24.",".64.1.59.",".98...3..","...8.3.2.","........6","...2759.."]
    #[[],[],[],[],[],[],[],[],[]]


    #probably helpful if this returns True/False on success/failure
    s = sudoku_solver.SudokuSolver()
    s.solveSudoku(puzzle)
    print "Count:" ,s.count
    print "\n*********************\n"
    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in puzzle]))


    #print contour_values

    #PRINT SOLUTION
    #we need to know which boxes were empty (x), and their coordinates so we can write the new value
    i = 0
    for x,y,w,h,num in contour_values:
        if num == "":
            #draw at x,y, using puzzle[i/9][i%9]
            cv2.putText(img,str(puzzle[i/9][i%9]),(x+w/4,y+h * 3/4),0,3,(0,255,0),5)
            #cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
        i += 1

    #cv2.imwrite("static/output2.png", img)
    #cv2.imshow('FINAL',img)
    #cv2.waitKey(0)

    return img2, img

#run(cv2.imread('static/sudoku3.png'))
