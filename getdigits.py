import os
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import tryk as digitdecoder

def image_process_tofindbox(ref):

    #ref = cv2.resize(ref,(int(640/3),int(480/3)))
 
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    #(thresh, ref) = cv2.threshold(ref, 127, 255, cv2.THRESH_BINARY)
    # ref = cv2.threshold(ref, 50, 255, 0)[1]
    ref = cv2.GaussianBlur(ref, (7, 7), 0)
    ref = cv2.bilateralFilter(ref, 11, 17, 17)

    # ref = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    # ref = cv2.morphologyEx(ref, cv2.MORPH_OPEN, ref)     

    # ref = cv2.Canny(ref, 50, 200, 255)
    ref = cv2.Canny(ref, 75, 150)

    #ref = cv2.bitwise_not(ref)    
    #cv2.imwrite("imgref.png", ref)


    # ref = cv2.threshold(ref, 0, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)[1]       
    # ref = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    # ref = cv2.morphologyEx(ref, cv2.MORPH_OPEN, ref)     
    # ref = cv2.Canny(ref, 50, 200, 255)



    return ref

def get_contours (img) :

    ref = img.copy()
    ref = image_process_tofindbox(ref)
    contours, hierarchy = cv2.findContours(ref, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = cv2.findContours(ref, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]

    plt.imshow(ref)
    plt.show()

    imgcntr = img.copy()
    cv2.drawContours(imgcntr,contours,-1,(0,255,0),2)
    plt.imshow(imgcntr)
    plt.show()
    
    return contours

def image_process_tofinddigit(ref):

    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)    
    
    #ref = cv2.GaussianBlur(ref, (3, 3), 0)
    ref = cv2.GaussianBlur(ref, (5, 5), 0)    
    
    # ref = cv2.threshold(ref, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # ref = cv2.bitwise_not(ref)        
    # kernel = np.ones((3,3), np.uint8)
    # ref = cv2.erode(ref, kernel, iterations=1)
       
    # ref = cv2.threshold(ref, 0, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)[1]       
    # ref = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    # ref = cv2.morphologyEx(ref, cv2.MORPH_OPEN, ref)     
    # ref = cv2.Canny(ref, 50, 200, 255)

    ref = cv2.bilateralFilter(ref, 11, 17, 17)
    ref = cv2.Canny(ref, 75, 150)


    return ref

def get_contours2 (i) :

    ref = i.copy()
    ref = image_process_tofinddigit(ref)    
    contours,hierarchy = cv2.findContours(ref, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plt.imshow(ref)
    plt.show()

    imgcntr = i.copy()
    cv2.drawContours(imgcntr,contours,-1,(0,255,0),2)

    plt.imshow(imgcntr)
    plt.show()
    
    return ref,contours



def get_digits(img):

    img = cv2.resize(img,(int(640/1.5),int(480/1.5)))
    data = np.array(img,dtype=np.uint8)

    contours = get_contours(img)
    #print("Contours: ",contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:20]

    #ref,contours = get_contours2(img)
    digitdecoder.train()

    output = []
    should = []
    for cnt in contours :
        
        x,y,w,h = cv2.boundingRect(cnt)
        x,y,w,h = x-2, y-2, w+4, h+4
        r = w/h
        l = w*h
        r0 = 2.5*0.80
        r1 = 2.5*1.5
        
        print("R:",r0,r,r1," L:",l)
        item_box = data[y:y+h,x:x+w]

        result = -1
        if l>500 and l<6000 and r>0.1 and r<0.4:
            result = digitdecoder.decode(item_box) 

        cv2.imwrite("found\\"+str(result)+"_"+str(l)+"_"+str(r)+"_"+str(hash((x,y,w,h)))+".png",item_box)

        # plt.imshow(item_box)
        # plt.show()

        if( r>=r0 and r<=r1 ) :

            print("Box found")
            data_box = data[y:y+h,x:x+w]

            plt.imshow(data_box)
            plt.show()
            
            ref, cnts_box = get_contours2(data_box)
            

            for digit_contour in cnts_box :
                
                #print("process : ",digit_contour)
                xd,yd,wd,hd = cv2.boundingRect(digit_contour)
                
                xd,yd,wd,hd = xd-2,yd-2,wd+4,hd+4

                ld = wd*hd    
                rd = wd/hd
                rd0 = 0.3
                rd1 = 1.1
                ad = 500
                print(rd,(hd*wd))
                
                digit_box = data_box[yd:yd+hd+2,xd:xd+wd-4]      
                result = digitdecoder.decode(item_box) 
                cv2.imwrite("found\\"+"d_"+str(result)+"_"+str(ld)+"_"+str(rd)+"_"+str(hash((xd,yd,wd,hd)))+".png",digit_box)

                if( rd > rd0 and rd < rd1 and (hd*wd) > ad ) :
                    
                    #print("X:",xd)
                    

                    plt.imshow(digit_box)
                    plt.show()

                    no = input("Input Digit Should be ? ")
                    
                    result = digitdecoder.decode(digit_box) 
                    
                    #decode_digit(digit_box)
                    output.append(result)
                    should.append(no)

                    if result==int(no):
                        print(" !!!!!!!!!!!!! Found: Digit: "+str(result))
                    else:
                        if int(no)>=0:
                            hfn = str(hash(fn))
                            sno = str(no)
                            cv2.imwrite("chars\\"+hfn+"_1_"+sno+".png", digit_box)


            break


    print("Digits found: ",output)
    print("Digits Should: ",should)

    return output


#fn = "data\\06755299.jpg"
#fn = "opencv_frame_3.png"
fn = "opencv_frame_5.png"
#fn = "token.jpeg"

img = cv2.imread(fn)
digits = get_digits(img)