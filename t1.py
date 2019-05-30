import os
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

def getFiles(folder):
    files = glob.glob(folder+"/*")
    return files

def getNumbersFilename(f):
    fx = f.split("/")
    fn = fx[1].split(".jpg")
    numbers = list(fn[0])
    return numbers



# ============================================================================    

def extract_chars(img):
    ref = img.copy()
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(ref, 50, 255, 0)[1]
    ref = cv2.GaussianBlur(ref, (5, 5), 0)
    #ref = cv2.Canny(ref, 50, 200, 255)
    ref = cv2.bitwise_not(ref)    
    cv2.imwrite("imgref.png", ref)
    contours = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    #print(contours)

    im = img.copy()
    cv2.drawContours(im,contours,-1,(0,255,0),2)
    cv2.imwrite("imgcont.png", im)

    #char_mask = np.zeros_like(ref)
    bounding_boxes = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        x,y,w,h = x-2, y-2, w+4, h+4
        print("w=",w,"h=",h)
        if w>40 and h>100 :
            bounding_boxes.append((x,y,w,h))


    print(bounding_boxes)

    characters = []
    for bbox in bounding_boxes:
        x,y,w,h = bbox
        char_image = ref[y:y+h,x:x+w]
        char_image_resized = cv2.resize(char_image
            , None
            , fx=3
            , fy=3
            , interpolation=cv2.INTER_CUBIC)
        characters.append(char_image_resized)

    #print(characters)

    return characters

# ============================================================================    

def output_chars(chars, labels):
    i=0
    for i, char in enumerate(chars):
        filename = "chars/%s.png" % i
        char = cv2.resize(char
            , None
            , fx=3
            , fy=3
            , interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(filename, char)

# ============================================================================    

def train() :
    #read image
    files = getFiles("data")
    for f in files :
        #read image      
        print("Reading ",f)  
        img = cv2.imread(f)
        digits = extract_chars(img)            
        numbers = getNumbersFilename(f)
        i=0
        for digit in digits :
            number = numbers[i]
            print("NUM = ",number)
            print("DIGIT = ",digit)
            filename_image = "chars/"+("".join(numbers))+"_"+str(number)+".png"
            print("Writing ",filename_image)
            cv2.imwrite(filename_image, digit)
            i=i+1
        
    # np.savetxt('digits.txt',np.array(x),delimiter=" ", fmt="%s")
    # np.savetxt('numbers.txt',np.array(y),delimiter=" ", fmt="%s")

def imageToBW(img):

    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
 
    return blackAndWhiteImage

def imageResize(img):

    w,h = np.shape(img)  
    #print("Shape image: ",(w,h))
    # r = 0.2
    # nw = int(w*r)
    # nh = int(h*r)
    nw,nh = (30,100)
    img_small = cv2.resize(img,(nw,nh))
    w,h = np.shape(img_small)  
    #print("Shape image_small: ",(w,h))

    return img_small

def loadImage(fn):

    print("Loading image: ",fn)
    img = cv2.imread(fn) 
    return readImage(img)

def readImage(img,isdebug=False):

    img = imageToBW(img)
    img_small = imageResize(img)
    #cv2.imwrite(fn+"_small.png", img_small)

    if isdebug:
        plt.imshow(img_small)
        plt.show()

    img_array = np.asarray(img_small,np.float32)
    img_array = img_array.reshape(-1)

    return img_array

def readImages() :
    files = getFiles("chars")
    #x=np.empty((0,10,3))
    # x=np.empty((100))
    x=[]
    y=[]
    for f in files :
        print("Reading ",f)
        fn = f.split("\\")[1]
        fn = fn.split(".png")[0]
        fn = fn.split("_")
        number = fn[2]    

        img_array = loadImage(f)
        
        #print("Number :"+number);
        #print("SHAPE : ",np.shape(img_array))
        #print(">>> :",img_array)
        
        x.append(img_array)
        y.append(number)

    xa = np.array(x,np.float32)
    ya = np.array(y,np.float32)
    np.savetxt('digits.txt',xa)
    np.savetxt('numbers.txt',ya)

    return xa,ya


def detect(img) :
    
    samples = np.loadtxt('digits.txt',np.float32)
    responses = np.loadtxt('numbers.txt',np.float32)
    responses = responses.reshape((responses.size,1))

   # model = cv2.KNearest()
    model = cv2.ml.KNearest_create()
    model.train(samples,cv2.ml.ROW_SAMPLE,responses)

    # ret, results, neighbours ,dist = knn.findNearest(newcomer, 3)
    # print( "result:  {}\n".format(results) )
    # print( "neighbours:  {}\n".format(neighbours) )
    # print( "distance:  {}\n".format(dist) )


#train()
#readImages()
#detect(0)


# if not os.path.exists("chars"):
#     os.makedirs("chars")

# img_digits = cv2.imread("digit1.png")
# #img_letters = cv2.imread("train_letters.png", 0)

# digits = extract_chars(img_digits)
# #letters = extract_chars(img_letters)

# DIGITS = [0, 6, 7 ,5, 5, 2, 9, 9]
# #LETTERS = [chr(ord('A') + i) for i in range(25,-1,-1)]

# output_chars(digits, DIGITS)
# #output_chars(letters, LETTERS)

# ============================================================================ 

