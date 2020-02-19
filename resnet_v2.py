import tensorflow as tf
from tensorflow.contrib import slim

from . import resnet_utils

resnet_arg_scope = resnet_utils.resnet_arg_scope


# 定义瓶颈函数（核心方法）
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections=None, scope=None):
    '''
    :param inputs:输入
    :param depth:Blocks类中的args
    :param depth_bottleneck:Blocks类中的args
    :param stride:Blocks类中的args
    :param outputs_collections:收集end_points的collection
    :param scope:unit的名称
    :return:
    '''
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        # 获取输入的最后一个维度，输出通道数
        depth_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
        # 对输入进行batch_borm，接着用relu进行预激活
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        #定义shortcut(直连的x)
        if depth == depth_in:
            # 如果残差单元输入通道和输出通道数一样，就对inputs进行降采样
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            # 如果残差单元输入通道与输出通道数不一样，就使用stride步长的1*1卷积改变其通道数，是的输入通道和输出通道数一样
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')
        # 定义残差
        # 第一步：3*3，stride=stride，输出通道数为depth_bottleneck的卷积
        # 第二步：3*3，stride=1，输出通道数为depth的卷积
        residual = slim.conv2d(preact, depth_bottleneck, [3, 3], stride=stride, scope='conv1')
        residual = slim.conv2d(residual, depth, [3, 3], stride=1, scope='conv2')
        output = shortcut + residual

        # 将结果添加到outputs_collections
        return utils.collect_named_outputs(outputs_collections, sc.name, output)


# 定义生成resnet_v2的主函数
def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=True,
              include_root_block=True,
              reuse=None,
              scope=None):

    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_point_collections = sc.original_name_scope + '_end_points'
        # 用slim.arg_scope将slim.conv2d bottleneck stack_blocks_dense 3个函数的参数outputs_collections设置为end_point_collections
        with slim.arg_scope([slim.conv2d, bottleneck, resnet_utils.stack_block_dense], outputs_collections=end_point_collections):
            net = inputs

            if include_root_block:
                # 根据include_root_block标记，创建resnet最前面一层的卷积神经网络
                with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                    net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            # 利用stack_blocks_dense将残差学习模块完成
            net = resnet_utils.stack_block_dense(net, blocks)
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')

            if global_pool:
                # 根据标记添加平均池化层
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)

            if num_classes is not None:
                # 根据是否有分类数，添加一个输出通道为num_classes的1*1卷积
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')

            # utils.convert_collection_to_dict将collection转化为dict
            end_points = utils.convert_collection_to_dict(end_point_collections)

            if num_classes is not None:
                # 添加一个softmax输出层
                end_points['prediction'] = slim.softmax(net, scope='prediction')

            return net, end_points


# 定义resnet_v2_18的生成方法
def resnet_v2_18(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_18'
                 ):
    # 设计18层的resnet
    # 四个blocks的units数量为2、2、2、2，总层数为（2+2+2+2）*2+2=18
    # 后3个blocks包含步长为2的层，总尺寸244/(4*2*2*2)=7 输出通道变为512
    blocks = [
        resnet_utils.Block('block1', bottleneck, [(64, 64, 1)] * 2),
        resnet_utils.Block('block2', bottleneck, [(128, 128, 2)] + [(128, 128, 1)]),
        resnet_utils.Block('block3', bottleneck, [(256, 256, 2)] + [(256, 256, 1)]),
        resnet_utils.Block('block4', bottleneck, [(512, 512, 2)] + [(512, 512, 1)])
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)


if __name__ == '__main__':
    input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input')
    with slim.arg_scope(resnet_arg_scope()) as sc:
        logits = resnet_v2_18(input)