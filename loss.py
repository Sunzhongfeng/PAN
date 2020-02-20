import tensorflow as tf
import itertools


def pan_loss(outputs, labels, training_masks, alpha=0.5, beta=0.25, reduction='mean'):
    """
    Implement PAN Loss.
    :param outputs: 预测输出
    :param labels: 真实标签
    :param training_masks:
    :param alpha: loss kernel 前面的系数
    :param beta: loss agg 和 loss dis 前面的系数
    :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
    """
    texts = outputs[:, :, :, 0]
    kernels = outputs[:, :, :, 1]
    gt_texts = labels[:, :, :, 0]
    gt_kernels = labels[:, :, :, 1]

    # 计算 agg loss 和 dis loss
    similarity_vectors = outputs[:, :, :, 2:]
    loss_aggs, loss_diss = agg_dis_loss(texts, kernels, gt_texts, gt_kernels, similarity_vectors)

    # 计算 text loss
    loss_texts = dice_loss(texts, gt_texts, training_masks)

    # 计算 kernel loss
    one = tf.ones_like(texts)
    zero = tf.zeros_like(texts)
    W = tf.where(Sn[5] >= 0.5, x=one, y=zero)  # 整个文本区域的mask
    loss_kernels = dice_loss(kernels*W, gt_kernels*W, selected_masks)

    # mean or sum
    if reduction == 'mean':
        loss_text = tf.reduce_mean(loss_texts)
        loss_kernel = tf.reduce_mean(loss_kernels)
        loss_agg = tf.reduce_mean(loss_aggs)
        loss_dis = tf.reduce_mean(loss_diss)
    elif self.reduction == 'sum':
        loss_text = tf.reduce_sum(loss_texts)
        loss_kernel = tf.reduce_sum(loss_kernels)
        loss_agg = tf.reduce_sum(loss_aggs)
        loss_dis = tf.reduce_sum(loss_diss)
    else:
        raise NotImplementedError

    loss_all = loss_text + alpha * loss_kernel + beta * (loss_agg + loss_dis)
    return loss_all, loss_text, loss_kernel, loss_agg, loss_dis


def agg_dis_loss(texts, kernels, gt_texts, gt_kernels, similarity_vectors, delta_agg, delta_dis):
    """
    计算 loss agg
    :param texts: 文本实例的分割结果 batch_size * (w*h)
    :param kernels: 缩小的文本实例的分割结果 batch_size * (w*h)
    :param gt_texts: 文本实例的gt batch_size * (w*h)
    :param gt_kernels: 缩小的文本实例的gt batch_size*(w*h)
    :param similarity_vectors: 相似度向量的分割结果 batch_size *(w*h) * 4
    :param delta_agg: 计算loss agg时的常量
    :param delta_dis: 计算loss dis时的常量
    :return:
    """
    batch_size = texts.shape[0]
    texts = texts.contiguous().reshape(batch_size, -1)
    kernels = kernels.contiguous().reshape(batch_size, -1)
    gt_texts = gt_texts.contiguous().reshape(batch_size, -1)
    gt_kernels = gt_kernels.contiguous().reshape(batch_size, -1)
    similarity_vectors = similarity_vectors.contiguous().view(batch_size, 4, -1)
    loss_aggs = []
    loss_diss = []
    for text_i, kernel_i, gt_text_i, gt_kernel_i, similarity_vector in zip(texts, kernels, gt_texts, gt_kernels,
                                                                           similarity_vectors):
        text_num = gt_text_i.max().item() + 1
        loss_agg_single_sample = []
        G_kernel_list = []  # 存储计算好的G_Ki,用于计算loss dis
        # 求解每一个文本实例的loss agg
        for text_idx in range(1, int(text_num)):
            # 计算 D_p_Ki
            single_kernel_mask = gt_kernel_i == text_idx
            if single_kernel_mask.sum() == 0 or (gt_text_i == text_idx).sum() == 0:
                # 这个文本被crop掉了
                continue
            # G_Ki, shape: 4
            G_kernel = similarity_vector[:, single_kernel_mask].mean(1)  # 4
            G_kernel_list.append(G_kernel)
            # 文本像素的矩阵 F(p) shape: 4* nums (num of text pixel)
            text_similarity_vector = similarity_vector[:, gt_text_i == text_idx]
            # ||F(p) - G(K_i)|| - delta_agg, shape: nums
            text_G_ki = (text_similarity_vector - G_kernel.reshape(4, 1)).norm(2, dim=0) - delta_agg
            # D(p,K_i), shape: nums
            D_text_kernel = tf.pow(tf.maximum(text_G_ki, tf.zeros_like(text_G_ki)), 2)
            # 计算单个文本实例的loss, shape: nums
            loss_agg_single_text = tf.log(D_text_kernel + 1)
            loss_agg_single_text = tf.reduce_mean(loss_agg_single_text)
            loss_agg_single_sample.append(loss_agg_single_text)
        if len(loss_agg_single_sample) > 0:
            loss_agg_single_sample = tf.reduce_mean(loss_agg_single_sample)
        else:
            loss_agg_single_sample = tf.zeros_like(texts)
        loss_aggs.append(loss_agg_single_sample)

        # 求解每一个文本实例的loss dis
        loss_dis_single_sample = 0
        for G_kernel_i, G_kernel_j in itertools.combinations(G_kernel_list, 2):
            # delta_dis - ||G(K_i) - G(K_j)||
            kernel_ij = delta_dis - (G_kernel_i - G_kernel_j).norm(2)
            # D(K_i,K_j)
            D_kernel_ij = tf.pow(tf.maximum(kernel_ij, tf.zeros_like(kernel_ij)), 2)
            loss_dis_single_sample += tf.log(D_kernel_ij + 1)
        if len(G_kernel_list) > 1:
            loss_dis_single_sample /= (len(G_kernel_list) * (len(G_kernel_list) - 1))
        else:
            loss_dis_single_sample = tf.zeros_like(texts)
        loss_diss.append(loss_dis_single_sample)
    return tf.stack(loss_aggs), tf.stack(loss_diss)


def dice_loss(y_true_cls, y_pred_cls, training_mask):
    '''
    dice loss
    :param y_true_cls: ground truth
    :param y_pred_cls: predict
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    dice = 2 * intersection / union
    loss = 1. - dice
    return loss
