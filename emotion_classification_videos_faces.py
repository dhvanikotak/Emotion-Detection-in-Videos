
# coding: utf-8

# In[3]:

import numpy as np
import cv2
import glob
from random import shuffle
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import datetime


def split_data(data, percentaje):    
    
    shuffle(data)
    train_n = int(percentaje*len(data))
    train, test = np.split(data, [train_n])

    s_train = zip(*train)
    s_test = zip(*test)
    
    samples_train = list(s_train[0])
    labels_train = list(s_train[1])

    samples_test = list(s_test[0])
    labels_test = list(s_test[1])
    
    return samples_train, labels_train, samples_test, labels_test


def draw_flow(img, flow, step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def calc_hist(flow):

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees = 1)
    
    q1 = ((0 < ang) & (ang <= 45)).sum()
    q2 = ((45 < ang) & (ang <= 90)).sum()
    q3 = ((90 < ang) & (ang <= 135)).sum()
    q4 = ((135 < ang) & (ang <= 180)).sum()
    q5 = ((180 < ang) & (ang <= 225)).sum()
    q6 = ((225 <= ang) & (ang <= 270)).sum()
    q7 = ((270 < ang) & (ang <= 315)).sum()
    q8 = ((315 < ang) & (ang <= 360)).sum()
    
    hist = [q1, q2, q3, q4 ,q5, q6, q7 ,q8]
    
    return (hist)


def process_video(fn, samples):

    video_hist = []
    hog_list = []
    sum_desc = []
    bins_n = 10

    cap = cv2.VideoCapture(fn)
    ret, prev = cap.read()
            
    prevgray = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()


    while True:
           
        ret, img = cap.read()
        
        if not ret : break
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prevgray,gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        prevgray = gray

        bins = np.hsplit(flow, bins_n)

        out_bins = []
        for b in bins:
            out_bins.append(np.vsplit(b, bins_n))

        frame_hist = []
        for col in out_bins:

            for block in col:
                frame_hist.append(calc_hist(block))
                                 
        video_hist.append(np.matrix(frame_hist) )

    # average per frame
    sum_desc = video_hist[0]
    for i in range(1, len(video_hist)):
        sum_desc = sum_desc + video_hist[i] 
    
    ave = sum_desc / len(video_hist)

    # max per bin
    maxx = np.amax(video_hist, 0)
    maxx = np.matrix(maxx)
    
    fn = fn.lower()

    if '_ha_' in fn: label = 1
    if '_su_' in fn: label = 2
    if '_sa_' in fn: label = 3
    if '_di_' in fn: label = 4
    if '_fe_' in fn: label = 5
    if '_an_' in fn: label = 6
            
    print label
    ave_desc = np.asarray(ave)
    a_desc = []
    a_desc.append(np.asarray(ave_desc, dtype = np.uint8).ravel())

    max_desc = np.asarray(maxx)
    m_desc = []
    m_desc = np.asarray(max_desc, dtype = np.uint8).ravel()

    return a_desc, label, m_desc


# In[4]:

if __name__ == '__main__':
        
    path = '/Users/dhvanikotak/Box Sync/CV Project/240x320/'
   # path = '/Users/soledad/Box Sync/Fall 15/I590 - Collective Intelligence/CV Project/240x320/'
    
#     folders = glob.glob(path+ "/*")
    folders = [path + 'Angry2', path + 'Surprised2', path + 'Disgusted2',
               path + 'Fear2', path + 'Sad2', path + 'Happy2']
    
    happy_data = []
    sad_data = []
    disgust_data = []
    fear_data = []
    surprise_data = []
    angry_data = []

    samples = 30
    a = datetime.datetime.now()
    for act in folders:
            fileList = glob.glob(act + "/*.mov")  
            print(len(fileList))

            for f in fileList:
                f = f.lower()
                print f
    
                if 'ha' in f:
                    video_desc, label, maxx = process_video(f, samples)
                
                    if (label) != 0 :
                        happy_data.append([video_desc[0], label, maxx])
                    
                if 'sa' in f:
                      
                    video_desc, label, maxx = process_video(f, samples)
                
                    if (label) != 0 :
                        sad_data.append([video_desc[0], label, maxx])
                
                if 'di' in f:
                       
                    video_desc, label, maxx = process_video(f, samples)
                
                    if (label) != 0 :
                        disgust_data.append([video_desc[0], label, maxx])
        
                if 'fe' in f:
                        
                    video_desc, label, maxx = process_video(f, samples)
                
                    if (label) != 0 :
                        fear_data.append([video_desc[0], label, maxx])
        
                if 'su' in f:
                       
                    video_desc, label, maxx = process_video(f, samples)
                
                    if (label) != 0 :
                        surprise_data.append([video_desc[0], label, maxx])
        
                if 'an' in f:
                        
                    video_desc, label, maxx = process_video(f, samples)
                
                    if (label) != 0 :
                        angry_data.append([video_desc[0], label, maxx])
                            

    b = datetime.datetime.now()
    
    print (b-a)


# In[2]:

import cPickle
percentaje = 0.7

clf = svm.SVC(kernel = 'rbf', C = 10, gamma = 0.0000001)

gnb = GaussianNB()
mnb = MultinomialNB()

svm = 0
nb1 = 0
nb2 = 0

# all_data = happy_data + sad_data + fear_data + surprise_data + disgust_data + angry_data
     
times = 10

for i in range(0,times):
    # happiness
    happy_samples_train = []
    happy_labels_train = []
    happy_samples_test = []
    happy_labels_test = []
    if len(happy_data) > 0:
        happy_samples_train, happy_labels_train, happy_samples_test, happy_labels_test = split_data(happy_data, percentaje)
      
    # sadness
    sad_samples_train = []
    sad_labels_train = []
    sad_samples_test = []
    sad_labels_test = []
    if len(sad_data) > 0:
        sad_samples_train, sad_labels_train, sad_samples_test, sad_labels_test = split_data(sad_data, percentaje)
   
    # fear
    fear_samples_train = []
    fear_labels_train = []
    fear_samples_test = []
    fear_labels_test = []
    if len(fear_data) > 0:
        fear_samples_train, fear_labels_train, fear_samples_test, fear_labels_test = split_data(fear_data, percentaje)
    
    # surprise
    surprise_samples_train = []
    surprise_labels_train = []
    surprise_samples_test = []
    surprise_labels_test = []
    if len(surprise_data) > 0:
        surprise_samples_train, surprise_labels_train, surprise_samples_test, surprise_labels_test = split_data(surprise_data, percentaje)
  
    # disgust
    disgust_samples_train = []
    disgust_labels_train = []
    disgust_samples_test = []
    disgust_labels_test = []
    if len(disgust_data) > 0:
        disgust_samples_train, disgust_labels_train, disgust_samples_test, disgust_labels_test = split_data(disgust_data, percentaje)
    
    # angrer
    angry_samples_train = []
    angry_labels_train = []
    angry_samples_test = []
    angry_labels_test = []
    if len(angry_data) > 0:
        angry_samples_train, angry_labels_train, angry_samples_test, angry_labels_test = split_data(angry_data, percentaje)
    
   
    
    train_set = happy_samples_train + sad_samples_train + fear_samples_train + surprise_samples_train + disgust_samples_train + angry_samples_train
    test_set = happy_samples_test + sad_samples_test + fear_samples_test + surprise_samples_test + disgust_samples_test + angry_samples_test
    labels_train = happy_labels_train + sad_labels_train + fear_labels_train + surprise_labels_train + disgust_labels_train + angry_labels_train
    labels_test = happy_labels_test + sad_labels_test + fear_labels_test + surprise_labels_test + disgust_labels_test + angry_labels_test 
     

    # train_set, labels_train, test_set, labels_test = split_data(all_data, percentaje)    

    clf.fit(train_set, labels_train)
    gnb.fit(train_set, labels_train)
    mnb.fit(train_set, labels_train)
    
    y_pred_g = gnb.predict(test_set)
    y_pred_m = mnb.predict(test_set)
    predicted = clf.predict(test_set) 
    
    err1 = (labels_test == predicted).mean()
    err2 = (labels_test == y_pred_g).mean()
    err3 = (labels_test == y_pred_m).mean()
        
    print 'accuracy svm: %.2f %%' % (err1*100), 'accuracy gnb: %.2f %%' % (err2*100), 'accuracy mnb: %.2f %%' % (err3*100)

#     folder = '/Users/soledad/Box Sync/Fall 15/I590 - Collective Intelligence/CV Project/Code/Emotion_Out/'

    folder = '/Users/dhvanikotak/Box Sync/CV Project/Code/Emotion_Out/'

    outfile = open(folder + str(i)+'train_set.pkl', 'wb')
    np.save(outfile, train_set)
    outfile.close()
    
    outfile = open(folder + str(i)+'test_set.pkl', 'wb')
    np.save(outfile, test_set)
    outfile.close()
    
    outfile = open(folder + str(i)+'labels_train.pkl', 'wb')
    np.save(outfile, labels_train)
    outfile.close()
    
    outfile = open(folder + str(i)+'labels_test.pkl', 'wb')
    np.save(outfile, labels_test)
    outfile.close()

    # save the classifier
    with open(folder + str(i)+'svm.pkl', 'wb') as fid:
        cPickle.dump(clf, fid)  
    fid.close()
    
    with open(folder + str(i)+'mnb.pkl', 'wb') as fid:
        cPickle.dump(mnb, fid)  
    fid.close()
    
    with open(folder + str(i)+'gnb.pkl', 'wb') as fid:
        cPickle.dump(gnb, fid)  
    fid.close()
    



# In[ ]:




# In[ ]:



