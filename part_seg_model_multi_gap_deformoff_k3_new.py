#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../'))

import tf_util
from transform_nets import input_transform_net
from gat_layers3 import attn_feature, multi_scale_gap_block


def get_model(point_cloud, input_label, is_training, cat_num, part_num,
              batch_size, num_point, weight_decay, bn_decay=None):

    k = 20
    num_heads = 4

    def multi_head_deformable_sampling(point_cloud, is_training, bn_decay, num_heads, k, scope):
        with tf.variable_scope(scope):
            head_outputs = []
            all_offsets = []
            local_features=[]
            for h in range(num_heads):
                with tf.variable_scope('head_%d' % h):
                    pc_exp = tf.expand_dims(point_cloud, 2)
                    net = tf_util.conv2d(pc_exp, 64, [1, 1], bn=True, is_training=is_training,
                                         scope='mlp1', bn_decay=bn_decay)
                    net = tf_util.conv2d(net, 32, [1, 1], bn=True, is_training=is_training,
                                         scope='mlp2', bn_decay=bn_decay)
                    offset = tf_util.conv2d(net, 3, [1, 1], activation_fn=tf.nn.tanh,
                                            scope='offset', bn_decay=bn_decay)
                    offset = offset * 0.05
                    offset = tf.squeeze(offset, [2])
                    all_offsets.append(offset)

                    pc_xyz = point_cloud[:, :, :3] + offset
                    pc_deformed = tf.concat([pc_xyz, point_cloud[:, :, 3:]], axis=-1)
                    
                    #dynamic_k=k+h*4
                    adj_matrix = tf_util.pairwise_distance(pc_deformed)
                    nn_idx = tf_util.knn(adj_matrix, k=k)

                    edge_feature, _,locals1= attn_feature(pc_deformed, 16, nn_idx, activation=tf.nn.elu,
                                                      in_dropout=0.6, coef_dropout=0.6, is_training=is_training,
                                                      bn_decay=bn_decay, layer='layer0_head%d' % h, k=k, i=0,
                                                      is_dist=True)
                    head_outputs.append(edge_feature)
                    local_features.append(locals1)

            
            multi_head_output = tf.concat(head_outputs, axis=-1)
            locals1= tf.concat(local_features, axis=-1)
            mean_offsets = tf.add_n(all_offsets) / num_heads  # [B, N, 3]
            pc_xyz = point_cloud[:, :, :3]  # [B, N, 3]
            return multi_head_output, mean_offsets, pc_xyz,locals1
    # --- 1. Multi-head Deformable Attention
    adj_matrix = tf_util.pairwise_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    n_heads = 1
    attns = []
    local_features = []
    for i in range(n_heads):
        edge_feature, coefs, locals = attn_feature(point_cloud, 16, nn_idx, activation=tf.nn.elu,
                                                   in_dropout=0.6,
                                                   coef_dropout=0.6, is_training=is_training, bn_decay=bn_decay,
                                                   layer='layer0', k=k, i=i, is_dist=True)
        attns.append(edge_feature)
        local_features.append(locals)
    neighbors_features = tf.concat(attns, axis=-1)
    neighbors_features = tf.concat([tf.expand_dims(point_cloud, -2), neighbors_features], axis=-1)
    print('neighbors_features1',neighbors_features.shape,'local',local_features)

    locals_max_transform = tf.reduce_max(tf.concat(local_features, axis=-1), axis=-2, keep_dims=True)
    
    print('locals_max_transform',locals_max_transform)
    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(neighbors_features, locals_max_transform, is_training, bn_decay, K=3, is_dist=True)

    point_cloud_transformed = tf.matmul(point_cloud, transform)


   

    # --- 1. Multi-head Deformable Attention
    multi_head_output, offsets, point_cloud_xyz,locals1 = multi_head_deformable_sampling(
        point_cloud_transformed, is_training, bn_decay, num_heads=num_heads, k=k, scope='multi_head_deformable')

    # --- 2. Feature Fusion
    neighbors_features = tf.concat([tf.expand_dims(point_cloud, -2), multi_head_output], axis=-1)
    #locals_max_transform = tf.reduce_max(multi_head_output, axis=-2, keepdims=True)

    #neighbors_features = tf.concat([tf.expand_dims(point_cloud, -2), neighbors_features], axis=-1)
   
    locals_max1 = tf.reduce_max(tf.concat(locals1, axis=-1), axis=-2, keep_dims=True)
    print('neighbors_features',neighbors_features.shape,'local',locals1)

    
    #print('locals_max_transform',locals_max_transform)


    # --- 4. GAPNet Backbone (Layer 1 and Layer 2)
    #adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
    #nn_idx = tf_util.knn(adj_matrix, k=k)

    #attns, local_features = [], []
    #for i in range(4):
        #edge_feature, _, locals = attn_feature(point_cloud_transformed, 16, nn_idx, activation=tf.nn.elu,
                                               #in_dropout=0.6, coef_dropout=0.6, #is_training=is_training,
                                               #bn_decay=bn_decay, layer='layer1', k=k, i=i, is_dist=True)
        #attns.append(edge_feature)
        #local_features.append(locals)
    #neighbors_features = tf.concat(attns, axis=-1)
    #neighbors_features = tf.concat([tf.expand_dims(point_cloud_transformed, -2), neighbors_features], #axis=-1)
    #locals_max1 = tf.reduce_max(tf.concat(local_features, axis=-1), axis=-2, keepdims=True)

    net = tf_util.conv2d(neighbors_features, 64, [1, 1], bn=True, is_training=is_training,
                         scope='gapnet1', bn_decay=bn_decay, is_dist=True)
    net = tf_util.conv2d(net, 64, [1, 1], bn=True, is_training=is_training,
                         scope='gapnet2', bn_decay=bn_decay, is_dist=True)
    net3 = tf_util.conv2d(net, 128, [1, 1], bn=True, is_training=is_training,
                          scope='gapnet3', bn_decay=bn_decay, is_dist=True)

    adj_matrix = tf_util.pairwise_distance(net3)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    attns, local_features = [], []
    for i in range(4):
        edge_feature, _, locals = attn_feature(net3, 128, nn_idx, activation=tf.nn.elu,
                                               in_dropout=0.6, coef_dropout=0.6, is_training=is_training,
                                               bn_decay=bn_decay, layer='layer2', k=k, i=i, is_dist=True)
        attns.append(edge_feature)
        local_features.append(locals)
    neighbors_features = tf.concat(attns, axis=-1)
    neighbors_features = tf.concat([tf.expand_dims(point_cloud_transformed, -2), neighbors_features], axis=-1)
    locals_max2 = tf.reduce_max(tf.concat(local_features, axis=-1), axis=-2, keepdims=True)
    
    
    

    net = tf_util.conv2d(neighbors_features, 128, [1, 1], bn=True, is_training=is_training,
                         scope='gapnet4', bn_decay=bn_decay, is_dist=True)
    net = tf_util.conv2d(net, 128, [1, 1], bn=True, is_training=is_training,
                         scope='gapnet5', bn_decay=bn_decay, is_dist=True)
    net6 = tf_util.conv2d(net, 512, [1, 1], bn=True, is_training=is_training,
                          scope='gapnet6', bn_decay=bn_decay, is_dist=True)

    # --- 5. Global Feature
    net = tf_util.conv2d(tf.concat([net3, net6, locals_max1, locals_max2], axis=-1), 1024, [1, 1],
                         bn=True, is_training=is_training, scope='gapnet8', bn_decay=bn_decay, is_dist=True)
    out_max = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='maxpool')

    # --- 6. Class Label Embedding
    label_feat = tf.reshape(input_label, [batch_size, 1, 1, cat_num])
    label_feat = tf_util.conv2d(label_feat, 64, [1, 1], bn=True, is_training=is_training,
                                scope='label_embed', bn_decay=bn_decay, is_dist=True)

    # --- 7. Concatenation and MLP Segmentation Head
    expand = tf.tile(tf.concat([out_max, label_feat], axis=-1), [1, num_point, 1, 1])
    concat = tf.concat([expand, net], axis=3)

    net = tf_util.conv2d(concat, 256, [1, 1], bn=True, is_training=is_training,
                         scope='seg/conv1', weight_decay=weight_decay, is_dist=True)
    net = tf_util.dropout(net, keep_prob=0.6, is_training=is_training, scope='seg/dp1') 
    net = tf_util.conv2d(net, 256, [1, 1], bn=True, is_training=is_training,
                         scope='seg/conv2', weight_decay=weight_decay, is_dist=True)
    net = tf_util.dropout(net, keep_prob=0.6, is_training=is_training, scope='seg/dp2')
    net = tf_util.conv2d(net, 128, [1, 1], bn=True, is_training=is_training,
                         scope='seg/conv3', weight_decay=weight_decay, is_dist=True)
    net = tf_util.conv2d(net, part_num, [1, 1], activation_fn=None, bn=False,
                         scope='seg/conv4', weight_decay=weight_decay, is_dist=True)

    seg_pred = tf.reshape(net, [batch_size, num_point, part_num])
    return seg_pred, offsets, point_cloud_xyz




def get_loss(seg_pred, seg_label, offsets=None, point_cloud_xyz=None, mode='train'):
    # Cross-entropy loss
    if mode == 'train':
        per_instance_seg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
             logits=seg_pred, labels=seg_label)
        seg_loss = tf.reduce_mean(per_instance_seg_loss)

    # Offset magnitude loss
        offset_l2 = tf.reduce_mean(tf.square(offsets))

    # Smoothness: difference between offset and its neighbors
        adj_matrix = tf_util.pairwise_distance(point_cloud_xyz)
        nn_idx = tf_util.knn(adj_matrix, k=8)
        neighbors = tf_util.get_neighbors(offsets, nn_idx)
        offsets_exp = tf.expand_dims(offsets, axis=2)
        smoothness_l2 = tf.reduce_mean(tf.square(offsets_exp - neighbors))

        lambda_mag = 0.001
    #mu_smooth = 0.01
        mu_smooth=0.001
        offset_reg_loss = lambda_mag * offset_l2 + mu_smooth * smoothness_l2

        total_loss = seg_loss + offset_reg_loss
        per_instance_seg_pred_res = tf.argmax(seg_pred, axis=2)
    else:
        per_instance_seg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
             logits=seg_pred, labels=seg_label)
        
        total_loss = tf.reduce_mean(per_instance_seg_loss)
        per_instance_seg_pred_res = tf.argmax(seg_pred, 2)

    return total_loss, per_instance_seg_loss, per_instance_seg_pred_res
