from PIL import Image
import numpy as np
import sys
import os
import csv
import cv2 as cv
import imutils
from seasons_script import * 

#TODO: you could also use seasons.py

# default format can be changed as needed
def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    labels = []
    names = []
    keywords = {"brightspring" : "0"} # keys and values to be changed as needed
    count = 0
    for root, dirs, files in os.walk(myDir, topdown=True):

           
            for name in files: #goes through all the sub directories to all teh files. 
           
                pathtofile = os.path.join(root, name)
              
                if name.endswith(format):
                     
                    fullName = os.path.join(root, name)
                    print(fullName, "fullname")
                    fileList.append(fullName)
                elif name.endswith("png"): #dealing with png... IDK Why this is needed honestly?
                    im1 = Image.open(pathtofile)
                    newname = name[:-3] + "jpg"
                    print(newname)
                    im1.save(os.path.join(root, newname))
                    os.remove(pathtofile)
                    print("removed", pathtofile)
                    print(os.path.join(root, newname))
                    fullName = os.path.join(root, newname)
                    fileList.append(fullName)
                else:
                     continue
                # could unpack and get facial recognition of the file 
                # print(fullName)
                # image_raw = cv.imread(fullName, cv.IMREAD_COLOR)  # reads into BGR
                  
                # orig_num_rows, orig_num_cols, _ = image_raw.shape          # cool underscore variable!
                # num_rows, num_cols, _ = image_raw.shape
              
                # image=cv.imread(pathtofile)
                # image = imutils.resize(fullName,width=250) #resize
                                
                # image = cv.imread(fullName, cv.IMREAD_COLOR)  # reads into BGR
                # orig_num_rows, orig_num_cols, _ = image.shape          # cool underscore variable!
                # num_rows, num_cols, _ = image.shape
               
                # # save it as numpy array here
                for keyword in keywords:
                    if keyword in name:
                        labels.append(keywords[keyword]) #label it
                    else:
                        continue
                names.append(name)
    return fileList, labels, names
    # return labels 
                

# load the filelist 

# load the original image 
# myFileList, labels, names  = createFileList('/content/')
myFileList, labels, names  = createFileList('dataset')
i = 0
for file in myFileList:
    print(file)
    img_file = Image.open(file) # don't really need this... or the lines below
    # img_file.show()
# get original image parameters...
    # width, height = img_file.size
    # format = img_file.format
    # mode = img_file.mode

# TODO: facial recognition 

    theface = facialrecognition(file) #read in teh file not opened img_file...
    #read with cv2
    fig, ax = plt.subplots()
    plt.imshow(cv2.cvtColor(theface, cv2.COLOR_BGR2RGB))
    ax.axis('off')  
    plt.show()
    print("got here")
    theface = imutils.resize(theface,width=250) #resize to be smaller  IN BGR #TODO: Make it smaller!
    im_rgb = cv2.cvtColor(theface, cv2.COLOR_BGR2RGB) #concert to RGB
    img = Image.fromarray(im_rgb)
    # img = Image.fromarray(theface.astype('uint8'), 'RGB')
    #TODO: CONVERT TO RGB
    img.save('facetouse.png') #YAY Correct!
    new_img_file = Image.open('facetouse.png')
    width, height = new_img_file.size
    print(width ,height)
    format = new_img_file.format
    mode = new_img_file.mode

# Make image Greyscale 
    #img = Image.fromarray(theface) 
    # img = Image.fromarray(theface.astype('uint8'), 'RGB')

    # img = img[:,:,::-1]

    img = new_img_file.convert('RGB') # figured out how to save the face into a numpy array with proper colors 

    # img.show() #uncomment if you want to see this?

    # img.save('newresult.png') #NOTE: this will create an image in documents but won't show up anywhere??

    # img = cv2.imread('newresult.png')
    # fig, ax = plt.subplots()
    # plt.imshow(img)
    # ax.axis('off')  
    # plt.show()
    # print("what")
# Save Greyscale values

    # TODO: change the array into the right dimensions

    # TODO: reshape
   
    value = np.asarray(img.getdata(), dtype=int)
    print(value.shape, "shape")
    # value = np.arange(187500).reshape((62500, 3))
    value = value.reshape((250, 250, 3)) #NOTE: add 3 dimension since it's a ... RGB :0
    

    
    value = value.flatten()
    
    value = np.append(value,labels[i])
    i +=1
    
    print(value)
    with open("data_test.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)


# test how to open