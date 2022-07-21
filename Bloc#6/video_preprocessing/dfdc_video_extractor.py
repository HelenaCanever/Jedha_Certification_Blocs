import pandas as pd
import cv2
import math
import os
from PIL import Image, ImageFilter
import face_recognition

# specify folder. Must contain metadata file with labels
videoFolder = "train_sample_videos"

#make subfolders for preprocessed videos
REAL_path = os.path.join(videoFolder, "extracted_REAL")
FAKE_path = os.path.join(videoFolder, "extracted_FAKE")
if not os.path.exists(REAL_path):
  os.mkdir(REAL_path)
if not os.path.exists(FAKE_path):
  os.mkdir(FAKE_path)


# build a loop for the folder
for videoName in os.listdir(videoFolder):
    #processes only video files
    if videoName.endswith(".mp4"):
        #make subfolder for frames
        sub_path = os.path.join(videoFolder, videoName.rstrip('.mp4'))
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)

        print(f"Extracting: {videoName}")
        #define subpath  of video to preprocess
        path = os.path.join(videoFolder, videoName)
        videocap = cv2.VideoCapture(path)        

        frameRate = videocap.get(5) #frame rate

        while(videocap.isOpened()):
            frameId = videocap.get(1) #current frame number
            ret, frame = videocap.read() 
            if (ret != True):
                break
            #we arbitrarily chose to extract 5 frames per second
            if (frameId % math.floor(frameRate/5)) == 0 and (frameId <= (frameRate*5)) : 
                filename = videoName.rstrip('.mp4') +  str(int(frameId)) + ".jpg"
                filepath = os.path.join(sub_path, filename)
                cv2.imwrite(filepath, frame) #create frame image
        videocap.release()

        n=0
        images=[] 
        for imagename in os.listdir(sub_path): #loop over extracted frames
            if imagename.endswith(".jpg"):
                path = os.path.join(sub_path, imagename)
                #face recognition
                image = face_recognition.load_image_file(path)
                #face coordinates in px
                face_locations = face_recognition.face_locations(image)
                #prevents code from breaking if no face is detected in frame
                try:
                    top, right, bottom, left = face_locations[0] #get location of first face detected, sometimes other people are in frame or false faces are detected
                except:
                    continue
            
                # calculate 10% padding
                pad = int((bottom-top)*0.1)
                #create image
                try:
                    face_image = image[(top-pad):(bottom+pad), (left-pad):(right+pad)]
                    pil_image = Image.fromarray(face_image)
                except:
                    face_image = image[(top):(bottom), (left):(right)]
                    pil_image = Image.fromarray(face_image)
                
                #resize and blur
                im = pil_image.resize((128, 128))
                im = im.filter(ImageFilter.GaussianBlur(radius = 0.5))
                im_path = os.path.join(sub_path, str(n) + ".png")
                im = im.save(im_path)
                images.append(str(n) + ".png")
                n+=1

        if len(images)>10: #check that at least 10 frames have been extracted
            frame = cv2.imread(os.path.join(sub_path, images[0]))
            height, width, layers = frame.shape

            #check image label
            metadata = pd.read_json(os.path.join(videoFolder, "metadata.json"))
            #create face video
            video_name_extracted =  videoName.rstrip('.mp4')+"_extracted"
            if metadata[videoName]['label'] == 'REAL':
                video = cv2.VideoWriter(os.path.join(REAL_path,video_name_extracted+".avi"), 0, 1, (width,height))
            else:
                video = cv2.VideoWriter(os.path.join(FAKE_path,video_name_extracted+".avi"), 0, 1, (width,height))
            for image in images:
                video.write(cv2.imread(os.path.join(sub_path, image)))

            cv2.destroyAllWindows()
            video.release()
        
print("All done")