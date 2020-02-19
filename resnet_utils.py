import collections
import tensorflow as tf

slim = tf.contrib.slim


# 定义Block类
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """
    定义一个Block类，有三个属性：
    'scope'： 命名空间
    'unit_fn'：单元函数
    'args'：参数
    """


# 定义相关函数
def subsample(inputs, factor, scope=None):
    """
    降采样方法
    :param inputs: 输入数据
    :param factor: 采样因子 1：不做修改直接返回 不为1：使用slim.max_pool2d降采样
    :param scope: 作用域
    :return:
    """
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, kernel_size=[1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    """
    定义卷积操作方法
    """
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', scope=scope)
    else:
        # 修整Inputs,对inputs进行补零操作
        padding_total = kernel_size - 1
        padding_beg = padding_total // 2
        padding_end = padding_total - padding_beg
        inputs = tf.pad(inputs, [[0, 0], [padding_beg, padding_end], [padding_beg, padding_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding='VALID', scope=scope)


# 定义堆叠block方法
@slim.add_arg_scope
def stack_block_dense(net, blocks, outputs_collections=None):
    """
    net:input
    blocks:Block的class列表
    outputs_collections:收集各个end_points的collections
    """
    for block in blocks:
        # 双层循环，遍历blocks，遍历res unit堆叠
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            # 用两个tf.variable_scope将残差学习单元命名为block1/unit_1的形式
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    # 利用第二层循环拿到block中的args,将其展开为depth,depth_bottleneck,strdie
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    # 使用残差学习单元的生成函数unit_fn，顺序的创建并连接所有的残差学习单元
                    net = block.unit_fn(net,
                                        depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
                    # 使用utils.collect_named_outputs将输出net添加到collection中
                    net = utils.collect_named_outputs(outputs_collections, sc.name, net)
    return net


# 定义resnet参数
def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.97,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    # 创建resnet通用的arg_scope
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS}
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc
