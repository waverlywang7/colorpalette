# libraries!
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import seaborn as sns

import imutils
import cv2
import colorsys

import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint

import statistics
from numpy import mean

    
from typing import Tuple
import numpy as np


# NOTE: using matplotlib showing stuff





def facialrecognition(imageinput):
# Read the fileurl and return the faces
    if isinstance(imageinput, np.ndarray): #testing if numpy image
        image = imutils.resize(imageinput,width=250)
    elif isinstance(imageinput, str): # if it's a string
        print("I got here")
        if imageinput[0:6] == 'https:': #if url    
            image =  imutils.url_to_image(imageinput)
            image = imutils.resize(image,width=250) # important step to resize bc the min size of face was small in comparison to some images before. :0
        else: #it's a file
            image=cv2.imread(imageinput)
            image = imutils.resize(image,width=250) #resize
    else:
        print("I got here instead")
        print(imageinput)
        print(type(imageinput))
        image=cv2.imread(imageinput)
        image = imutils.resize(image,width=250) #resize

    fig, ax = plt.subplots()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis('off')  


    ax.set_title('Color') 
    plt.show()

    image_faces_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    # fig, ax = plt.subplots()
    # ax.imshow(image_faces_gray,cmap="gray")
    # ax.axis('off')  
    # ax.set_title('Gray') 
    # plt.show()
    faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        image_faces_gray,       # this is the input image
        scaleFactor=1.05,       # this is the scale-resolution for detecting faces
        minNeighbors=1,         # this is how many nearby-detections are needed to ok a face
        minSize=(10,10),        # this is the minimum size for a face
        flags = cv.CASCADE_SCALE_IMAGE, 
        # (standard)

    )
    print(f"Found {len(faces)} faces!")

    maximum = 0  # keep track of the confidence levels. Highest confidence level is the one with biggest bounding box.
    maxFace = 0
    for i, face in enumerate(faces):
        x,y,w,h = face
        if w*h > maximum:
            maximum = w*h
            maxFace = i
        print(f"face {i}: {face}")
    print(f"the face that is probably a face is {maxFace} with a size of {maximum}")
        
    

    image_faces_drawn_rgb = image.copy()  # copy onto which we draw the bounding boxes for the faces

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # note that this draws on the color image!
        cv.rectangle(image_faces_drawn_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2) 

    print(f"Drawn! How does it do?")

    LoFi = []  # list of Face images

   
    for (x, y, w, h) in faces: 
        # note that this draws on the color image!
        face = image[y:y+h,x:x+h,:]  #, (x, y), (x+w, y+h), (0, 255, 0), 2)  
        LoFi.append( face )
        value = w*h
       


    # show the image!
    fig, ax = plt.subplots(figsize=(6,6)) 
    plt.imshow(cv2.cvtColor(image_faces_drawn_rgb,cv2.COLOR_BGR2RGB))
    ax.axis('off')  
    ax.set_title('FACES???') 
    plt.show()

    # printing the graph of LoFi
    print(f"There are {len(LoFi)} faces detected - they are held in the list 'LoFi'")
    print(f"Here are some of them...")

    
    fig, ax = plt.subplots(3,3)  # this means ax will be a 3x3 numpy array of axes!
    #ax[0,0].imshow(LoFi[0])
    #ax[0,0].imshow(cv2.resize(LoFi[0],dsize=(20,20)))
    ax[0,0].imshow(cv2.cvtColor(LoFi[0],cv2.COLOR_BGR2RGB))
    ax[0,0].axis('off') 
    if len(LoFi) >= 2: 
        ax[0,1].imshow(cv2.cvtColor(LoFi[1],cv2.COLOR_BGR2RGB))
        ax[0,1].axis('off')
    if len(LoFi) >= 3: 
        ax[0,2].imshow(cv2.cvtColor(LoFi[2],cv2.COLOR_BGR2RGB))
        ax[0,2].axis('off')
    if len(LoFi) >= 4: 
        ax[1,0].imshow(cv2.cvtColor(LoFi[3],cv2.COLOR_BGR2RGB))
        ax[1,0].axis('off')
    if len(LoFi) >= 5: 
        ax[1,1].imshow(cv2.cvtColor(LoFi[4],cv2.COLOR_BGR2RGB))
        ax[1,1].axis('off')
    if len(LoFi) >= 6: 
        ax[1,2].imshow(cv2.cvtColor(LoFi[5],cv2.COLOR_BGR2RGB))
        ax[1,2].axis('off')
    if len(LoFi) >= 7: 
        ax[2,0].imshow(cv2.cvtColor(LoFi[6],cv2.COLOR_BGR2RGB))
        ax[2,0].axis('off')
    if len(LoFi) >= 8: 
        ax[2,1].imshow(cv2.cvtColor(LoFi[7],cv2.COLOR_BGR2RGB))
        ax[2,1].axis('off')
    if len(LoFi) >= 9: 
        ax[2,2].imshow(cv2.cvtColor(LoFi[8],cv2.COLOR_BGR2RGB)) 
        ax[2,2].axis('off')
    plt.show()

    return LoFi[maxFace]

def extractSkin(image):
  # Taking a copy of the image
  img =  image.copy()
  # Converting from BGR Colours Space to HSV
  img =  cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
  
  # Defining HSV Threadholds
  lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
  upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
  
  # Single Channel mask,denoting presence of colours in the about threshold
  skinMask = cv2.inRange(img,lower_threshold,upper_threshold)
  
  # Cleaning up mask using Gaussian Filter
  skinMask = cv2.GaussianBlur(skinMask,(3,3),0)
  
  # Extracting skin from the threshold mask
  skin  =  cv2.bitwise_and(img,img,mask=skinMask)
  
  # Return the Skin image
  return cv2.cvtColor(skin,cv2.COLOR_HSV2BGR)

def removeBlack(estimator_labels, estimator_cluster):
  
  
  # Check for black
  hasBlack = False
  
  # Get the total number of occurance for each color
  occurance_counter = Counter(estimator_labels)

  
  # Quick lambda function to compare to lists
  compare = lambda x, y: Counter(x) == Counter(y)
   
  # Loop through the most common occuring color
  for x in occurance_counter.most_common(len(estimator_cluster)):
    
    # Quick List comprehension to convert each of RBG Numbers to int
    color = [int(i) for i in estimator_cluster[x[0]].tolist() ]
    
  
    
    # Check if the color is [0,0,0] that if it is black 
    if compare(color , [0,0,0]) == True:
      # delete the occurance
      del occurance_counter[x[0]]
      # remove the cluster 
      hasBlack = True
      estimator_cluster = np.delete(estimator_cluster,x[0],0)
      break
      
   
  return (occurance_counter,estimator_cluster,hasBlack)
    
def getColorInformation(estimator_labels, estimator_cluster,hasThresholding=False):
  
  # Variable to keep count of the occurance of each color predicted
  occurance_counter = None
  
  # Output list variable to return
  colorInformation = []
  
  
  #Check for Black
  hasBlack =False
  
  # If a mask has be applied, remove th black
  if hasThresholding == True:
    
    (occurance,cluster,black) = removeBlack(estimator_labels,estimator_cluster)
    occurance_counter =  occurance
    estimator_cluster = cluster
    hasBlack = black
    
  else:
    occurance_counter = Counter(estimator_labels)
 
  # Get the total sum of all the predicted occurances
  totalOccurance = sum(occurance_counter.values()) 
  
 
  # Loop through all the predicted colors
  for x in occurance_counter.most_common(len(estimator_cluster)):
    
    index = (int(x[0]))
    
    # Quick fix for index out of bound when there is no threshold
    index =  (index-1) if ((hasThresholding & hasBlack)& (int(index) !=0)) else index
    
    # Get the color number into a list
    color = estimator_cluster[index].tolist()
    
    # Get the percentage of each color
    color_percentage= (x[1]/totalOccurance)
    
    #make the dictionay of the information
    colorInfo = {"cluster_index":index , "color": color , "color_percentage" : color_percentage }
    
    # Add the dictionary to the list
    colorInformation.append(colorInfo)
    
      
  return colorInformation 


def extractDominantColor(image,number_of_colors=5,hasThresholding=False):
  
  # Quick Fix Increase cluster counter to neglect the black(Read Article) 
  if hasThresholding == True:
    number_of_colors +=1
  
  # Taking Copy of the image
  img = image.copy()
  
  # Convert Image into RGB Colours Space
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  
  # Reshape Image
  img = img.reshape((img.shape[0]*img.shape[1]) , 3)
  
  #Initiate KMeans Object
  estimator = KMeans(n_clusters=number_of_colors, random_state=0)
  
  # Fit the image
  estimator.fit(img)
  
  # Get Colour Information
  colorInformation = getColorInformation(estimator.labels_,estimator.cluster_centers_,hasThresholding)
  return colorInformation
  
def plotColorBar(colorInformation):
  #Create a 500x100 black image
  color_bar = np.zeros((100,500,3), dtype="uint8")
  
  top_x = 0
  for x in colorInformation:    
    bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

    color = tuple(map(int,(x['color'])))
  
    cv2.rectangle(color_bar , (int(top_x),0) , (int(bottom_x),color_bar.shape[0]) ,color , -1)
    top_x = bottom_x
  return color_bar

def prety_print_data(color_info):
    for x in color_info:
        print(pprint.pformat(x))
        print()


def skinextractor(imageinput):

    if isinstance(imageinput, np.ndarray): #testing if numpy image
        image = imutils.resize(imageinput,width=250)
    elif isinstance(imageinput, str): # if it's a string
        if imageinput[0:6] == 'https:': #if url    
            image =  imutils.url_to_image(imageinput)
        else: #it's a file
            image=cv2.imread(imageinput)

    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.show()

    
    # Apply Skin Mask
    skin = extractSkin(image)

    plt.imshow(cv2.cvtColor(skin,cv2.COLOR_BGR2RGB))
    plt.show()



    # Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of colors 
    dominantColors = extractDominantColor(skin,hasThresholding=True)




    #Show in the dominant color information
    print("Color Information")
    # prety_print_data(dominantColors)


    #Show in the dominant color as bar
    print("Color Bar")
    colour_bar = plotColorBar(dominantColors)
    plt.axis("off")
    plt.imshow(colour_bar)
    plt.show()
    return dominantColors
    
def getskincolor(numpyarray):
    #given a numpy array get skin color

    
    numpyarray= imutils.resize(numpyarray,width=250)
    

    color_info = skinextractor(numpyarray)

    r,g,b = color_info[0]["color"]    #just choose the dominant color 
    percent = color_info[0]["color_percentage"]
    h,s,v = colorsys.rgb_to_hsv(r,g,b)

 
    if v >= 236:
        return "fair", (h,s,v)
    elif 225 <= v < 236:
        return "light", (h,s,v)
    elif 201 <= v < 225: 
        return "medium", (h,s,v)
    elif 130 <= v< 201: 
        return "tan", (h,s,v)
    elif 116 <= v < 130:
        return "dark"
    elif v < 116:
        return "deep", (h,s,v)

def newwarmorcool(numpyarray):
    #takes in numpyarray and then outputs if it's warm or cool for the colors.
    # NOTE: I used the range of 0 to 20 which came from the skin extraction min and max threshold values for h
    # I would then divide this 0 to 20 range by choosing a point 
    skincolor, (h,s,v) = getskincolor(numpyarray)


    if skincolor == "fair":
        mincool = 0
        maxcool = .07
        minwarm = .07
        maxwarm = 20
    
        if h <= maxcool:
            print("you are cool for", skincolor, "skin")
            norm =  ((h)-mincool)/(maxcool-mincool)
            print(1-norm, "cool")
            return 1-norm, "cool"

        elif h > maxcool:
            print("you are warm for ", skincolor, "skin") # have to create a range within 0 and 20. tear tear
            norm =  ((h)-minwarm)/(maxwarm-minwarm)
            print(norm, "cool")
            return norm, "cool" #how far is it from the divide between warm and cool?
    

    elif skincolor == "light":
        mincool = .97
        maxcool = 20
        minwarm = 0
        maxwarm = .97
        if h <= maxwarm:
            print("you are warm for ", skincolor, "skin")
            norm =  ((h)-minwarm)/(maxwarm-minwarm)
            print(1-norm, "warm")
            return 1-norm, "warm"

        elif h > maxwarm:
            print("you are cool for", skincolor, " skin") # have to create a range within 0 and 20. tear tear
            norm =  ((h)-mincool)/(maxcool-mincool)
            print(norm, "cool")
            return norm, "cool" #how far is it from the divide between warm and cool?
    
        
    elif skincolor == "medium":
        mincool = 0
        maxcool = .07
        minwarm = .07
        maxwarm = 20
    

        if h <= maxcool:
            print("you are cool for ", skincolor, "skin")
            norm =  ((h)-mincool)/(maxcool-mincool)
            print(1-norm, "cool")
            return 1-norm, "cool"

        elif h > maxcool:
            print("you are warm for ", skincolor, "skin") # have to create a range within 0 and 20. tear tear
            norm =  ((h)-minwarm)/(maxwarm-minwarm)
            return norm, "warm" #how far is it from the divide between warm and cool?
    elif skincolor == "tan": #all tan people are warm 
        mincool = 0
        maxwarm = 20
    
      
        return ((h)-mincool)/(maxwarm-mincool), "warm"
    
    elif skincolor == "dark":
        mincool = 0
        maxcool = .027
        minwarm = .027
        maxwarm = 20
    
        if h <= maxcool:
            print("you are cool for ", skincolor, "skin")
            norm =  ((h)-mincool)/(maxcool-mincool)
            print(1-norm, "cool")
            return 1-norm, "cool"

        elif h > maxcool:
            print("you are warm for ", skincolor, " skin") # have to create a range within 0 and 20. tear tear
            norm =  ((h)-minwarm)/(maxwarm-minwarm)
            print(norm, "warm" )
            return norm, "warm" #how far is it from the divide between warm and cool?
    elif skincolor == "deep":
        mincool = 0
        maxcool = .96
        minwarm = .96
        maxwarm = 20
    
        if h <= maxcool:
            print("you are cool for ", skincolor, "skin")
            norm =  ((h)-mincool)/(maxcool-mincool)
            print(1-norm,"cool")
            return 1-norm,"cool"

        elif h > maxcool:
            print("you are warm for ", skincolor, " skin") # have to create a range within 0 and 20. tear tear
            norm =  ((h)-minwarm)/(maxwarm-minwarm)
            print(norm, "warm" )
            return norm, "warm" #how far is it from the divide between warm and cool?
    
def newdarkorlight(numpyarray):


        #calls skin colors
        skincolor, (h,s,v) = getskincolor(numpyarray)
        print("This person has", skincolor, "skin")
        if skincolor == "fair" or skincolor == "light": #fair or light... are always light and see how light they are
            if skincolor == "fair":
        
                minv = 236 #dark 
                maxv = 255 #light

            elif skincolor == "light":
                minv = 225 
                maxv = 236
            normalizedvalue = ((v)-minv)/(maxv-minv)
        
            # return how much of a fair person are they?
            print(normalizedvalue, "light")
            return normalizedvalue, "light" #the amount that it's light. the greater it is the more light it is



        elif skincolor == "medium" or skincolor == "tan" : #tan or medium people can be dark or light for their skin tone. 
            if skincolor == "medium":
                minv = 201
                maxv = 224
            elif skincolor == "tan":
                minv = 130
                maxv = 201
            halfway = (minv + maxv)/2
            if v >= minv and v <= halfway: 
                normalizedvalue = ((v)-minv)/(halfway-minv)
                # if you're halfway is the maxvalue... then we want to measure how far away are you from halfway instead of distance from inv so do 1-halfway
                print("this person is dark for ", skincolor, "skin")
                print(1- normalizedvalue, "dark")
                return 1- normalizedvalue, "dark"

            elif v > halfway and v <= maxv:
                normalizedvalue = ((v)-halfway)/(maxv-halfway)
                print("this person is light for ", skincolor, "skin")
                print(normalizedvalue, "light")
                return normalizedvalue, "light"

        
        elif skincolor == "dark" or "deep": #deep and dark can be only dark. 
            if skincolor == "dark":
                minv =  116
                maxv =  130
            elif skincolor == "deep":
                minv = 116
                maxv = 80
            normalizedvalues = ((v-minv)/(maxv-minv))
            print(1-normalizedvalues, "dark")
            return 1-normalizedvalues, "dark" #1 - because low values are dark
    

def newclearorsoft(numpyarray):
    # given numpy array, output a 

    color_info = extractDominantColor(numpyarray,number_of_colors=5,hasThresholding=True)

    #Show in the dominant color information
    vlist = []
    whitelist = []
    for i in range(0,len(color_info)):
        r,g,b  = color_info[i]["color"]
        h,s,v = colorsys.rgb_to_hsv(r,g,b)

        if v >= 255-15 and v<= 255: 
            whitelist += [color_info[i]]

        if v != 255:
            vlist += [v]

    for white in whitelist:
        print("removing", white)
        color_info.remove(white)
        print(color_info, "yay")

    print("Color Bar")
    print(len(color_info), "after")
    colour_bar = plotColorBar(color_info)
    plt.axis("off")
    plt.imshow(colour_bar)
    plt.show()
   
    std1 = statistics.stdev(vlist)/mean(vlist)
    if std1 <= .47:
        min = 0
        max = .47
        value = (std1 - min)/(max-min)
        print(1-value, "soft")
        return 1-value, "soft" #most soft is 0 and least soft is .47
    elif std1 >= .47:
        min = .47
        max = 1
        value = (std1 - min)/(max-min) #least clear is .47 and most is 1
        return value, "clear"



COLOR_SEASON_DICT = {
    # spring
    ("clear", "warm"): "bright spring",
    ("warm", "clear"): "true spring",
    ("light", "warm"): "light spring",
    # summer
    ("light", "cool"): "light summer",
    ("cool", "muted"): "true summer",
    ("muted", "cool"): "soft summer",
    # autumn
    ("muted", "warm"): "soft autumn",
    ("warm", "muted"): "true autumn",
    ("dark", "warm"): "dark autumn",
    # winter
    ("dark", "cool"): "dark winter",
    ("cool", "clear"): "true winter",
    ("clear", "cool"): "bright winter"
}

def match_characteristics_to_season(primary_characteristic: str,
                                    secondary_characteristic: str) -> str:
    return COLOR_SEASON_DICT[
        (primary_characteristic, secondary_characteristic)]


def get_primary_and_secondary_characteristics(
        imageinput
) -> Tuple[str, str]:
    # takes in an imageinput
    print("GETTING FACE")
    face = facialrecognition(imageinput)
    print("GETTING HUE")

    huescore, hue = newwarmorcool(face)
    print("GETTING SAT")

    satscore, sat  = newclearorsoft(face)
    print("GETTING VAL")
    valscore, val = newdarkorlight(face)

    #TODO: just have the score already be calculated! And out of 100. 
    hsvlist = [[huescore, hue] , [satscore,sat], [valscore,val]]
    primarychar = max(hsvlist)[1]
    primarytuple = max(hsvlist)
    print(hue, sat, val, huescore, satscore, valscore)
    if "warm" == primarychar or "cool" == primarychar: 
        secondarychar = sat 
      
     
    else:
        secondarychar = hue

    return primarychar, secondarychar, hue, huescore, sat, satscore, val, valscore
    
    


def identify_color_season(imageinput): #can be a url, fileimage, or numpyarray?

        primary, secondary, hue, huescore, sat, satscore, val, valscore = get_primary_and_secondary_characteristics(imageinput) # take in a fileurl or fileimage. 
        color_season = match_characteristics_to_season(primary, secondary)

        return (f"(hue: {hue, huescore}, sat: {sat, satscore}, val: {val, valscore}), primary: {primary} , secondary: {secondary} so this person is a {color_season}!")
