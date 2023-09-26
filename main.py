import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import gait_nn
import cv2
from PIL import Image
from human_pose_nn import HumanPoseIRNetwork
import math

mpl.use('TKAgg')
tf.compat.v1.disable_eager_execution()
net_pose = HumanPoseIRNetwork()
net_gait = gait_nn.GaitNetwork(recurrent_unit = 'GRU', rnn_layers = 1)
net_pose.restore('../gait-recognition-master/models/MPII+LSP.ckpt')
#net_pose.restore('../gait-recognition-master/models/Human3.6m(1).ckpt')#导入检查点
net_gait.restore('../gait-recognition-master/models/H3.6m-GRU-1.ckpt')#导入检查点
img_batch = None

def angle(v1, v2):
  dx1 = v1[2] - v1[0]
  dy1 = v1[3] - v1[1]
  dx2 = v2[2] - v2[0]
  dy2 = v2[3] - v2[1]
  angle1 = math.atan2(dy1, dx1)
  angle1 = int(angle1 * 180/math.pi)
  # print(angle1)
  angle2 = math.atan2(dy2, dx2)
  angle2 = int(angle2 * 180/math.pi)
  # print(angle2)
  if angle1*angle2 >= 0:
    included_angle = abs(angle1-angle2)
  else:
    included_angle = abs(angle1) + abs(angle2)
    if included_angle > 180:
      included_angle = 360 - included_angle
  return included_angle

def getSquareLen(v):
    x1 = v[0]
    y1 = v[1]
    x2 = v[2]
    y2 = v[3]
    return (x1-x2)**2+(y1-y2)**2

def showProbabilities(src_img,img_batch):
    #img = imread(img)
    re_img = np.array(Image.fromarray(src_img).resize((299, 299)))
    img_batch_onTime = np.expand_dims(re_img, 0)# 使用np.append(img1,img2,axis=0)堆叠筛选后的结果

    y, x, a = net_pose.estimate_joints(img_batch_onTime)  #
    y, x, a = np.squeeze(y), np.squeeze(x), np.squeeze(a)


    joint_names = [
        'right ankle ',
        'right knee ',
        'right hip',
        'left hip',
        'left knee',
        'left ankle',
        'pelvis',
        'thorax',
        'upper neck',
        'head top',
        'right wrist',
        'right elbow',
        'right shoulder',
        'left shouVlder',
        'left elbow',
        'left wrist'
    ]

    # Print probabilities of each estimation  绘制骨骼图
    # for i in range(16):
    #     print('%s: %.02f%%' % (joint_names[i], a[i] * 100))
    colors = ['b','g','r','c','m','m','y','k','w']
    v1 = [x[6], y[6], x[7], y[7]]#y
    v2 = [x[8], y[8], x[9], y[9]]#w
    v3 = [x[1],y[1],x[2],y[2]]#b
    v4 = [x[4],y[4],x[5],y[5]]#m
    lenv1 = getSquareLen(v1) #黄
    lenv2 = getSquareLen(v2) #白
    if angle(v1,v2) > 60 or lenv1<lenv2 or angle(v1,v4)<60 or angle(v1,v3)>60 or lenv1<100:
        return re_img,img_batch
    else:
        if img_batch is None:
            img_batch = img_batch_onTime
        else:
            img_batch = np.append(img_batch,img_batch_onTime,axis=0)
            #print(lenv1)
    # for i in [0,1,2,3,4,6,7,8]:
    #     plt.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color=colors[i], linewidth=5)
    return re_img,img_batch


videoCapture = cv2.VideoCapture('test2.mp4')
if not videoCapture.open('test2.mp4'):
    print("can not open file")
    exit(1)
i = 0
count=5
number = 1
plt.ion()
while True:
    success, frame = videoCapture.read()
    if not success:
        break
    r,g,b=cv2.split(frame)
    frame = cv2.merge([b,g,r])
    i += 1
    print(number)
    number+=1
    if (i % count == 0):
        plt.clf()
        img,img_batch = showProbabilities(frame,img_batch)
        plt.imshow(img)
        plt.show()
        plt.pause(0.0001)

    if not success:
        print('video is all read')
        break

plt.ioff()
plt.show()
spatial_features = net_pose.feed_forward_features(img_batch)#把reshape过的原图是带入ResNet提取特征
heatMap = net_pose.heat_maps(img_batch)
print(heatMap[0].shape)
identification_vector = net_gait.feed_forward(spatial_features)#线性化特征，得出特征向量
#print(identification_vector)#打印特征向量

# plt.savefig('images/img_pose.jpg')