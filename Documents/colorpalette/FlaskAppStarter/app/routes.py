# Authors: CS-World Domination Summer19 - JG
try:
    from flask import render_template, redirect, url_for, request, send_from_directory, flash
except:
    print("Make sure to pip install Flask twilio")
from app import app
import os

from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

from app import piglatin 
# from app import seasons
from app.seasons import *


try:
    from PIL import Image
    import PIL.ImageOps
    from numpy import asarray
    import numpy
                

except:
    print("Make sure to pip install Pillow")

# Home page, renders the index.html template
@app.route('/index',methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def index():
        if request.method == 'POST':
        # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # If user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            # if the image is valid, do the following
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Create a path to the image in the upload folder, save the upload
                # file to this path
                save_old=(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                file.save(save_old)
                # Take the image, make a new one that is inverted
                img = Image.open(save_old)
 
                
                rbg_img = img.convert('RGB')
                rbg_img = asarray(rbg_img) #convert to a numpy array!
                # inverted_image = PIL.ImageOps.invert(rbg_img)

                print("GETTING FACE")
                # season = skinextractor(rbg_img)
                face = facialrecognition(rbg_img) #niceee face is a numpy array
                # let's put code for facial recognition here
                face = asarray(face)
                # if numpy.size(face, axis=None): #if the resulting face is None. 
                #     face = rbg_img #don't use facial recognition and just use the image. 

                # huescore = newwarmorcool(face)  
                # hue = huescore
                # valscore,val = newdarkorlight(face)  
                # satscore,sat = newclearorsoft(face) 

             
                primary, secondary, hue, huescore, sat, satscore, val, valscore = get_primary_and_secondary_characteristics(face) # take in a fileurl or fileimage. 
                color_season = match_characteristics_to_season(primary, secondary)

               
                skinextracted = extractSkin(face)
                actualskincolor, rgb = getskincolor(face)
                color_info = skinextractor(skinextracted)

                colour_bar = plotColorBar(color_info)


                # season = skinextractor(face)
                img = Image.fromarray(face, "RGB") #perhaps put before 776

                another_img = Image.fromarray(skinextracted, "RGB")
                
                color_img = Image.fromarray(colour_bar)

                save_new=(os.path.join(app.config['UPLOAD_FOLDER'], 'new_'+filename))
                filename2 = "skin" + filename
                save_new2 = (os.path.join(app.config['UPLOAD_FOLDER'], filename2))
                filename3 = "color.png"
                save_new3 = (os.path.join(app.config['UPLOAD_FOLDER'], filename3))
             
                # turn numpy array into 
                # face = cv2.imread(save_new)
                try:
                    img.save(save_new)
                    another_img.save(save_new2)
                    color_img.save(save_new3)
                except AttributeError:
                    print("Couldn't save image {}".format(face))

              


                # Render template with inverted picture
                # rt = render_template('imageResults.html', filename='new_'+filename)
                rt = render_template('seasonResults.html', huescore = huescore, hue = hue, valscore = valscore,
                                                             val = val, satscore = satscore, sat = sat, actualskincolor = actualskincolor,
                                                            primary = primary, secondary = secondary, color_season = color_season,
                                                             filename2 = filename2, filename3 = filename3, filename1 = 'new_'+filename)
                # rt = render_template('seasonResults.html',  huescore = huescore, hue=hue,filename1 = 'new_'+filename,filename2= filename2)
                return rt
        return render_template('index.html', title= 'Home')

# Pig latin page, when we click translate, moves to text result page
@app.route('/text',methods=['GET','POST'])
def text():
    if request.method == 'POST':
        old_text = request.form['text']
        new_text = piglatin.pig_translate(old_text)
        return render_template('textResults.html', old_text=old_text, new_text=new_text)
    return render_template('text.html', title='Home')

# Used for uploading pictures
@app.route('/<filename>')
def get_file(filename):
    return send_from_directory('static',filename)

# Image uploading page, 
@app.route('/image', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # if the image is valid, do the following
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Create a path to the image in the upload folder, save the upload
            # file to this path
            save_old=(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save(save_old)
            # Take the image, make a new one that is inverted
            img = Image.open(save_old)
            rbg_img = img.convert('RGB')
            
            inverted_image = PIL.ImageOps.invert(rbg_img)
            save_new=(os.path.join(app.config['UPLOAD_FOLDER'], 'new_'+filename))
            inverted_image.save(save_new)
            # Render template with inverted picture
            rt = render_template('imageResults.html', filename='new_'+filename)
            return rt
    return render_template('image.html')

# allowed image types 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['ALLOWED_EXTENSIONS']=ALLOWED_EXTENSIONS

# is file allowed to be uploaded?
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
