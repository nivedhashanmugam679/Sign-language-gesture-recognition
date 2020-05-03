import cv2
import numpy as np
import os

# Create the directory structure
if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/test")
    os.makedirs("data/train/0")
    os.makedirs("data/train/1")
    os.makedirs("data/train/2")
    os.makedirs("data/train/3")
    os.makedirs("data/train/4")
    os.makedirs("data/train/5")
    os.makedirs("data/train/A1")
    os.makedirs("data/train/C1")
    os.makedirs("data/train/D1")
    os.makedirs("data/train/H1")
    os.makedirs("data/train/L1")
    os.makedirs("data/train/M1")
    os.makedirs("data/train/P1")
    os.makedirs("data/train/R1")
    os.makedirs("data/train/U1")
    os.makedirs("data/test/0")
    os.makedirs("data/test/1")
    os.makedirs("data/test/2")
    os.makedirs("data/test/3")
    os.makedirs("data/test/4")
    os.makedirs("data/test/5")
    os.makedirs("data/test/A1")
    os.makedirs("data/test/C1")
    os.makedirs("data/test/D1")
    os.makedirs("data/test/H1")
    os.makedirs("data/test/L1")
    os.makedirs("data/test/M1")
    os.makedirs("data/test/P1")
    os.makedirs("data/test/R1")
    os.makedirs("data/test/U1")
 
# Train or test 
mode = 'train'
directory = 'data/'+mode+'/'

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Getting count of existing images
    count = {'zero': len(os.listdir(directory+"/0")),
             'one': len(os.listdir(directory+"/1")),
             'two': len(os.listdir(directory+"/2")),
             'three': len(os.listdir(directory+"/3")),
             'four': len(os.listdir(directory+"/4")),
             'five': len(os.listdir(directory+"/5"))}
             'a': len(os.listdir(directory+"/A1"))}
             'c': len(os.listdir(directory+"/C1"))}
             'd': len(os.listdir(directory+"/D1"))}
             'h': len(os.listdir(directory+"/H1"))}
             'l': len(os.listdir(directory+"/L1"))}
             'm': len(os.listdir(directory+"/M1"))}
             'p': len(os.listdir(directory+"/P1"))}
             'r': len(os.listdir(directory+"/R1"))}
             'u': len(os.listdir(directory+"/U1"))}
    
    # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : "+mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ZERO : "+str(count['zero']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "ONE : "+str(count['one']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "TWO : "+str(count['two']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "THREE : "+str(count['three']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "FOUR : "+str(count['four']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "FIVE : "+str(count['five']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "A : "+str(count['a']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "C : "+str(count['c']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "D : "+str(count['d']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "H : "+str(count['h']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "L : "+str(count['l']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "M : "+str(count['m']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "P : "+str(count['p']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "R : "+str(count['r']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "U : "+str(count['u']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64)) 
 
    cv2.imshow("Frame", frame)
    
    #_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    #kernel = np.ones((1, 1), np.uint8)
    #img = cv2.dilate(mask, kernel, iterations=1)
    #img = cv2.erode(mask, kernel, iterations=1)
    # do the processing after capturing the image!
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", roi)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory+'0/'+str(count['zero'])+'.jpg', roi)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory+'1/'+str(count['one'])+'.jpg', roi)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory+'2/'+str(count['two'])+'.jpg', roi)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory+'3/'+str(count['three'])+'.jpg', roi)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory+'4/'+str(count['four'])+'.jpg', roi)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory+'5/'+str(count['five'])+'.jpg', roi)
    if interrupt & 0xFF == ord('A1'):
        cv2.imwrite(directory+'A1/'+str(count['a'])+'.jpg', roi)
    if interrupt & 0xFF == ord('C1'):
        cv2.imwrite(directory+'C1/'+str(count['c'])+'.jpg', roi)
    if interrupt & 0xFF == ord('D1'):
        cv2.imwrite(directory+'D1/'+str(count['d'])+'.jpg', roi)
    if interrupt & 0xFF == ord('H1'):
        cv2.imwrite(directory+'H1/'+str(count['h'])+'.jpg', roi)
    if interrupt & 0xFF == ord('L1'):
        cv2.imwrite(directory+'L1/'+str(count['l'])+'.jpg', roi)
    if interrupt & 0xFF == ord('M1'):
        cv2.imwrite(directory+'M1/'+str(count['m'])+'.jpg', roi)
    if interrupt & 0xFF == ord('P1'):
        cv2.imwrite(directory+'P1/'+str(count['p'])+'.jpg', roi)
    if interrupt & 0xFF == ord('R1'):
        cv2.imwrite(directory+'R1/'+str(count['r'])+'.jpg', roi)
    if interrupt & 0xFF == ord('U1'):
        cv2.imwrite(directory+'U1/'+str(count['u'])+'.jpg', roi)
    
cap.release()
cv2.destroyAllWindows()
"""
d = "old-data/test/0"
newd = "data/test/0"
for walk in os.walk(d):
    for file in walk[2]:
        roi = cv2.imread(d+"/"+file)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        cv2.imwrite(newd+"/"+file, mask)     
"""
