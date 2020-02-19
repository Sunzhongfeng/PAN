import tensorflow as tf
from tensorflow.contrib import slim
import resnet_v2


#使用双线性插值缩放图片
def unpool(inputs, rate):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*rate,  tf.shape(inputs)[2]*rate])


def FPEM_FFM(C):

    fpem_repeat = 2
    conv_out = 128
    c2 = C['C2']
    c3 = C['C3']
    c4 = C['C4']
    c5 = C['C5']
    # reduce channel
    c2 = slim.conv2d(c2, num_outputs=conv_out, kernel_size=[1, 1], activation_fn=tf.nn.relu,
                     normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,)
    c3 = slim.conv2d(c3, num_outputs=conv_out, kernel_size=[1, 1], activation_fn=tf.nn.relu,
                     normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,)
    c4 = slim.conv2d(c4, num_outputs=conv_out, kernel_size=[1, 1], activation_fn=tf.nn.relu,
                     normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,)
    c5 = slim.conv2d(c5, num_outputs=conv_out, kernel_size=[1, 1], activation_fn=tf.nn.relu,
                     normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,)
    fpems = []
    for i in range(fpem_repeat):
        fpems.append(FPEM(c2, c3, c4, c5, conv_out))

    # FPEM
    for i, fpem in enumerate(fpems):
        c2, c3, c4, c5 = fpem(c2, c3, c4, c5)
        if i == 0:
            c2_ffm = c2
            c3_ffm = c3
            c4_ffm = c4
            c5_ffm = c5
        else:
            c2_ffm += c2
            c3_ffm += c3
            c4_ffm += c4
            c5_ffm += c5

    # FFM
    c5 = unpool(c5_ffm, c2_ffm.size()[-2:])
    c4 = unpool(c4_ffm, c2_ffm.size()[-2:])
    c3 = unpool(c3_ffm, c2_ffm.size()[-2:])
    Fy = tf.concat([c2_ffm, c3, c4, c5], axis=-1)
    pred = slim.conv2d(Fy, 6, 1) #进行1X1卷积处理
    return pred


def FPEM(c2, c3, c4, c5, in_channels=128):
    # up阶段
    c4 = tf.nn.separable_conv2d(upsample_add(c5, c4), [3, 3, in_channels, in_channels],
                                     [1, 1, in_channels, in_channels], 1, padding='SAME')
    c3 = tf.nn.separable_conv2d(upsample_add(c4, c3), [3, 3, in_channels, in_channels],
                               [1, 1, in_channels, in_channels], 1, padding='SAME')
    c2 = tf.nn.separable_conv2d(upsample_add(c3, c2), [3, 3, in_channels, in_channels],
                               [1, 1, in_channels, in_channels], 1, padding='SAME')

    # down 阶段
    c3 = tf.nn.separable_conv2d(upsample_add(c3, c2), [3, 3, in_channels, in_channels],
                               [1, 1, in_channels, in_channels], 2, padding='SAME')
    c4 = tf.nn.separable_conv2d(upsample_add(c4, c3), [3, 3, in_channels, in_channels],
                               [1, 1, in_channels, in_channels], 2, padding='SAME')
    c5 = tf.nn.separable_conv2d(upsample_add(c5, c4), [3, 3, in_channels, in_channels],
                               [1, 1, in_channels, in_channels], 2, padding='SAME')
    return c2, c3, c4, c5


def upsample_add(x, y):
    return unpool(x, y.size()[2:]) + y
