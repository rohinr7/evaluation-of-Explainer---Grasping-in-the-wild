import cv2 
import pandas as pd
import os
import numpy as np
from scipy import interpolate
import pathlib

def extractframes(pathforvideo,savepath):

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Path to video file 
    vidObj = cv2.VideoCapture(pathforvideo) 
    # Used as counter variable 
    count = 0
    # checks whether frames were extracted 
    success = 1
    while success: 
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
        if success:
            # Saves the frames with frame-count 
            cv2.imwrite(f"{savepath}/frame%d.jpg" % count, image) 
            count += 1

def plotgaze(pathforcsv, extractframepath):
    data = pd.read_csv(pathforcsv)
    
    frame=0
    # gx = np.array(data.iloc[frame-1:frame+1,3])
    gx = np.array(data.iloc[0:frame+1,3])
    print(gx)    

    frames =  os.listdir(extractframepath)

    for frame in range(len(frames)):
        imagepath = f"frame{frame}.jpg"
        img = cv2.imread(f"{extractframepath}/{imagepath}")
        # img = cv2.resize(img, (227, 227), 
        #        interpolation = cv2.INTER_LINEAR)
        
        if frame % 2 == 0:
            if frame == 0:
                gx = np.array(data.iloc[0:frame+1,3])
                gy = np.array(data.iloc[0:frame+1,4])
                # g3x = np.array(data.iloc[0:frame+1,5])
                # g3y = np.array(data.iloc[0:frame+1,6])
                # g3z = np.array(data.iloc[0:frame+1,7]) 
            else:
                gx = np.array(data.iloc[frame-1:frame,3])
                gy = np.array(data.iloc[frame-1:frame,4])
                # g3x = np.array(data.iloc[frame-1:frame,5])
                # g3y = np.array(data.iloc[frame-1:frame,6])
                # g3z = np.array(data.iloc[frame-1:frame,7]) 


        cv2.circle(img, (int(gx),int(gy)), 30, (0,255,0), 3) 
        # cv2.circle(img, (int(g3x),int(g3y)), 30, (0,0,255), 3) 
        cv2.imshow("image", img) 
        cv2.waitKey(0)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break
    cv2.destroyAllWindows()




def PlotGaze2(pathforcsv, extractframepath):
    
    #read the frame
    data = pd.read_csv(pathforcsv , delimiter=',')                   
    #gx and gy will be in a array 
    gx = np.array(data.iloc[:,2])
    gy = np.array(data.iloc[:,3])

    frames =  os.listdir(extractframepath)
    num_frames = len(frames)
    print(num_frames)
    gaze_times = np.linspace(0, 1, len(gx))

    # Evenly distribute the frames over the same time range
    frame_times = np.linspace(0, 1, num_frames)

    # Perform linear interpolation for gx and gz
    interp_gx = interpolate.interp1d(gaze_times, gx, kind='linear')
    interp_gy = interpolate.interp1d(gaze_times, gy, kind='linear')

    # Interpolate gaze coordinates to match frame times
    interpolated_gx = interp_gx(frame_times)
    interpolated_gy = interp_gy(frame_times)
    

    file_name = pathlib.PureWindowsPath(pathforcsv).stem
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_writer = cv2.VideoWriter(f"{file_name}.mp4", fourcc , 25 ,(1920,1080))
    #looping through the frames
    for frame in range(len(frames)):
        imagepath = f"frame{frame}.jpg"
        img = cv2.imread(f"{extractframepath}/{imagepath}")

        cv2.circle(img, (int(interpolated_gx[frame]),int(interpolated_gy[frame])), 30, (255,0,0), 3) 
        
        video_writer.write(img)
        cv2.imshow("image", img) 
        cv2.waitKey(11)
            
    cv2.destroyAllWindows() 

        


if __name__ == '__main__': 
    #plotgaze(r"C:\Users\rohin\Desktop\New folder (2)\trdp\dataset\Bowl\Bowl\BowlPlace1Subject1\BowlPlace1Subject1.csv",
             #r"C:\Users\rohin\Desktop\New folder (2)\trdp\dataset\Bowl\Bowl\BowlPlace1Subject1\extract images")

    PlotGaze2(r"C:\Users\rohin\Desktop\New folder (2)\trdp\dataset\Bowl\Bowl\BowlPlace1Subject4\BowlPlace1Subject4.txt",r"C:\Users\rohin\Desktop\New folder (2)\trdp\dataset\Bowl\Bowl\BowlPlace1Subject4\extractimages")
    #extractframes(r"C:\Users\rohin\Desktop\New folder (2)\trdp\dataset\Bowl\Bowl\BowlPlace1Subject4\BowlPlace1Subject4.mp4", r"C:\Users\rohin\Desktop\New folder (2)\trdp\dataset\Bowl\Bowl\BowlPlace1Subject4\extractimages" )        