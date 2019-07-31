import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from array import array
import tensorflow as tf
import scipy.misc
import cv2
import math
import numpy as np
sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph('./models/atan/model_atan.ckpt.meta')
saver.restore(sess, "./models/atan/model_atan.ckpt")

graph = tf.get_default_graph()
predicted_angle = graph.get_tensor_by_name("predicted_angle:0")
true_image = graph.get_tensor_by_name("true_image:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")

img = cv2.imread('/Users/mohdsaquib/downloads/autopilot/steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0


#read data.txt
xs = []
ys = []
with open("/Users/mohdsaquib/downloads/autopilot/driving_dataset/data.txt") as f:
    for line in f:
        xs.append("/Users/mohdsaquib/downloads/autopilot/driving_dataset/driving_dataset/" + line.split()[0]) 
        ys.append(float(line.split()[1]) * scipy.pi / 180)
        
#get number of images
num_images = len(xs)


i = math.ceil(num_images*0.7)
print("Starting frame of video:" +str(i))

while(cv2.waitKey(10) != ord('q')):
    full_image = cv2.imread("/Users/mohdsaquib/downloads/autopilot/driving_dataset/" + str(i) + ".jpg")
    cv2.imshow('Frame Window', full_image)
    image = ((cv2.resize(full_image[-150:], (200, 66)) / 255.0))
    degrees = predicted_angle.eval(feed_dict={true_image: [image], keep_prob: 1.0})[0][0] * 180.0 / scipy.pi
    print("Steering angle: " + str(degrees) + " (pred)\t" + str(ys[i]*180/scipy.pi) + " (actual)")
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()
sess.close()
