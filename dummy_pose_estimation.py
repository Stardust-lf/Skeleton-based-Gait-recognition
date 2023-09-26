import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# from scipy.misc import imresize, imread
import tensorflow.compat.v1 as tf
from human_pose_nn import HumanPoseIRNetwork
import imageio
tf.compat.v1.disable_eager_execution()
net_pose = HumanPoseIRNetwork()
net_pose.restore('models/MPII+LSP.ckpt')

# img = imread('images/dummy.jpg')
# img = imresize(img, [299, 299])
img = imageio.imread('images/dummy.jpg')
img = tf.image.resize(img, [299, 299])
img_batch = np.expand_dims(img, 0)

y, x, a = net_pose.estimate_joints(img_batch)
y, x, a = np.squeeze(y), np.squeeze(x), np.squeeze(a)

# Create image
colors = ['r', 'r', 'b', 'm', 'm', 'y', 'g', 'g', 'b', 'c', 'r', 'r', 'b', 'm', 'm', 'c']
for i in range(16):
    if i < 15 and i not in {5, 9}:
        plt.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color = colors[i], linewidth = 5)

plt.imshow(img)
# plt.savefig('images/dummy_pose.jpg')
