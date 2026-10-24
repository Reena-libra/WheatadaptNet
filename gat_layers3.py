#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tf_util
import numpy as np


def attn_feature(input_feature, output_dim, neighbors_idx, activation, in_dropout=0.0, coef_dropout=0.0, is_training=None, bn_decay=None, layer='', k=20, i=0, is_dist=False):
    batch_size = input_feature.get_shape()[0].value
    num_dim = input_feature.get_shape()[-1].value

    input_feature = tf.squeeze(input_feature)
    if batch_size == 1:
        input_feature = tf.expand_dims(input_feature, 0)

    input_feature = tf.expand_dims(input_feature, axis=-2)

    #print('input_feature',input_feature.shape)


    # if in_dropout != 0.0:
    #     input = tf.nn.dropout(input, 1.0 - in_dropout)

    new_feature = tf_util.conv2d_nobias(input_feature, output_dim, [1, 1], padding='VALID', stride=[1, 1], bn=True,
                                        is_training=is_training, scope=layer + '_newfea_conv_head_' + str(i),
                                        bn_decay=bn_decay, is_dist=is_dist)

    #print('new f',new_feature.shape)

    neighbors = tf_util.get_neighbors(input_feature, nn_idx=neighbors_idx, k=k)
    #print('nrighbors',neighbors)
    input_feature_tiled = tf.tile(input_feature, [1, 1, k, 1])
    #print('input_feature_tiled',input_feature_tiled)
    edge_feature = input_feature_tiled - neighbors
    #print('edge',edge_feature)
    edge_feature = tf_util.conv2d(edge_feature, output_dim, [1, 1], padding='VALID', stride=[1, 1],
                               bn=True, is_training=is_training, scope=layer + '_edgefea_' + str(i), bn_decay=bn_decay, is_dist=is_dist)

    #print('edgef',edge_feature.shape)
    self_attention = tf_util.conv2d(new_feature, 1, [1, 1], padding='VALID', stride=[1, 1], bn=True,
                                  is_training=is_training, scope=layer+'_self_att_conv_head_'+str(i), bn_decay=bn_decay, is_dist=is_dist)
    
    #print('selfa',self_attention.shape)
    neibor_attention = tf_util.conv2d(edge_feature, 1, [1, 1], padding='VALID', stride=[1, 1], bn=True,
                                    is_training=is_training, scope=layer+'_neib_att_conv_head_'+str(i), bn_decay=bn_decay, is_dist=is_dist)

    print('natt',neibor_attention.shape)
    logits = self_attention + neibor_attention
    logits = tf.transpose(logits, [0, 1, 3, 2])

    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))
    # coefs = tf.ones_like(coefs)
    #
    # if coef_dropout != 0.0:
    #     coefs = tf.nn.dropout(coefs, 1.0 - coef_dropout)


    vals = tf.matmul(coefs, edge_feature)

    if is_dist:
        ret = activation(vals)
    else:
        ret = tf.contrib.layers.bias_add(vals)
        ret = activation(ret)

    print('final',ret.shape,coefs.shape,edge_feature.shape)
    return ret, coefs, edge_feature



def multi_scale_gap_block(point_cloud, k_list, is_training, bn_decay, scope, is_dist=True):    ###we are changing output dim of attn fetaure from 64 to 16
    """
    Apply multiple GAPLayers at different k values and fuse their outputs.
    :param point_cloud: input tensor [B, N, F]
    :param k_list: list of k values, e.g., [10, 20, 30]
    :return: fused output [B, N, C]
    """
    outputs = []
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for k in k_list:
            adj_matrix = tf_util.pairwise_distance(point_cloud)
            nn_idx = tf_util.knn(adj_matrix, k=k)
            edge_feature, _, _ = attn_feature(point_cloud, 64, nn_idx,
                                              activation=tf.nn.elu,
                                              in_dropout=0.6,
                                              coef_dropout=0.6,
                                              is_training=is_training,
                                              bn_decay=bn_decay,
                                              layer=f'gap_k{k}',
                                              k=k,
                                              i=0,
                                              is_dist=is_dist)
            outputs.append(edge_feature)  # [B, N, 64]

        # Concatenate multi-scale outputs
        multi_scale = tf.concat(outputs, axis=-1)  # [B, N, 3×64 = 192]
        #multi_scale = channel_attention_block(multi_scale, scope='channel_attention')  # [B, N, 192]
    return multi_scale


def multi_scale_gap_block_new(point_cloud, k_list, is_training, bn_decay, scope, is_dist=True): 
    """
    Apply multiple GAPLayers at different k values and fuse their outputs with reduced dimensionality.
    
    :param point_cloud: input tensor [B, N, C]
    :param k_list: list of k values (e.g., [10, 20, 30])
    :param bottleneck_dim: output feature dimension after bottleneck (e.g., 64)
    :return: fused output [B, N, bottleneck_dim]
    """
    outputs = []
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for k in k_list:
            adj_matrix = tf_util.pairwise_distance(point_cloud)
            nn_idx = tf_util.knn(adj_matrix, k=k)
            edge_feature, _, _ = attn_feature(point_cloud, 64, nn_idx,
                                              activation=tf.nn.elu,
                                              in_dropout=0.6,
                                              coef_dropout=0.6,
                                              is_training=is_training,
                                              bn_decay=bn_decay,
                                              layer=f'gap_k{k}',
                                              k=k,
                                              i=0,
                                              is_dist=is_dist)
            outputs.append(edge_feature)  # Each: [B, N, 64]

        # Concatenate multi-scale outputs: [B, N, 64 * len(k_list)]
        multi_scale = tf.concat(outputs, axis=-1)  # e.g., [B, N, 192] for 3 scales
        print(multi_scale.shape)
        # Bottleneck to reduce feature dimension
        #multi_scale = tf.expand_dims(multi_scale, axis=2)  # [B, N, 1, C]
        #print(multi_scale.shape)
        reduced = tf_util.conv2d(multi_scale, 64, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=True, is_training=is_training,
                                 scope='bottleneck_reduce', bn_decay=bn_decay,
                                 is_dist=is_dist)
        print(reduced.shape)
        #reduced = tf.squeeze(reduced, axis=2)  # [B, N, 64]
        print(reduced.shape)
        return reduced





def multi_scale_gap_block2(point_cloud, k_list, is_training, bn_decay, scope, is_dist=True):
    num_point = point_cloud.get_shape()[1].value
    outputs, local_features = [], []

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for i, k in enumerate(k_list):
            adj_matrix = tf_util.pairwise_distance(point_cloud)
            nn_idx = tf_util.knn(adj_matrix, k=k)

            layer_name = f"{scope}_gap_k{k}"
            with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                edge_feature, coefs, locals = attn_feature(
                    point_cloud, 64, nn_idx, activation=tf.nn.elu,
                    in_dropout=0.6, coef_dropout=0.6,
                    is_training=is_training, bn_decay=bn_decay,
                    layer=layer_name, k=k, i=0, is_dist=is_dist
                )
                outputs.append(edge_feature)
                local_features.append(locals)

        # Concatenate attention features
        multi_scale = tf.concat(outputs, axis=-1)  # [B, N, 64 * len(k_list)]

        # Fuse with locals_max (pooled over neighbors)
       ## Concatenate local edge features from all scales (along neighbor dimension)
        local_concat = tf.concat(local_features, axis=-2)  # [B, N, total_k, 64]
        print('lo',local_concat.shape)
       # Apply max pooling across neighbors to get local summary
        locals_max = tf.reduce_max(local_concat, axis=-2)  # [B, N, 64]
        print('lo',locals_max.shape)
        multi_scale = tf.squeeze(multi_scale, axis=2)
        print('multi_scale',multi_scale.shape)
        fused = tf.concat([multi_scale, locals_max], axis=-1)  # [B, N, C_total]
        fused = tf.expand_dims(fused, axis=2)
        print('fused',fused.shape)
        
        #fused = channel_attention_block(fused, scope=scope + '_se')

        # Optional projection to reduce dimensionality
        #used = tf_util.conv1d(fused, 128, 1, padding='VALID', bn=True,
                               #is_training=is_training, scope=scope + '_fuse_proj', bn_decay=bn_decay)
        
        

        return fused  # No need to return locals_max separately

    
    
def channel_attention_block(inputs, scope, reduction_ratio=16):
    """
    Squeeze-and-Excitation (SE) channel attention block.

    Args:
        inputs: Tensor of shape [B, N, C]
        scope: variable scope name
        reduction_ratio: compression factor for the bottleneck

    Returns:
        Tensor of shape [B, N, C] after channel-wise reweighting
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        channels = inputs.get_shape()[-1].value  # C

        # Step 1: Global average pooling → [B, C]
        squeeze = tf.reduce_mean(inputs, axis=1)  # [B, C]

        # Step 2: Bottleneck (FC → ReLU → FC → Sigmoid)
        excitation = tf.layers.dense(squeeze, units=channels // reduction_ratio,
                                     activation=tf.nn.relu, name='fc1')
        excitation = tf.layers.dense(excitation, units=channels,
                                     activation=tf.nn.sigmoid, name='fc2')  # [B, C]

        # Step 3: Reweight channels → reshape & scale
        excitation = tf.expand_dims(excitation, axis=1)  # [B, 1, C]
        scaled = inputs * excitation  # [B, N, C] * [B, 1, C] → broadcast

        return scaled
    
    
def multi_scale_gap_block3(point_cloud, k_list, is_training, bn_decay, scope, is_dist=True):
    """
    Enhanced Multi-Scale GAP block with scale attention and optional local fusion.
    Args:
        point_cloud: [B, N, C]
        k_list: list of K values (e.g., [10, 20, 30])
        Returns:
            fused: [B, N, D]
    """
    num_point = point_cloud.get_shape()[1].value
    scale_features = []
    local_features = []

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for i, k in enumerate(k_list):
            adj_matrix = tf_util.pairwise_distance(point_cloud)
            nn_idx = tf_util.knn(adj_matrix, k=k)

            layer_name = f"{scope}_gap_k{k}"
            with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                edge_feature, _, local = attn_feature(
                    point_cloud, 64, nn_idx, activation=tf.nn.elu,
                    in_dropout=0.6, coef_dropout=0.6,
                    is_training=is_training, bn_decay=bn_decay,
                    layer=layer_name, k=k, i=0, is_dist=is_dist
                )
                scale_features.append(edge_feature)      # [B, N, 64]
                local_features.append(local)             # [B, N, k, 64]

        # Stack scale features → [B, N, C, len(k_list)]
        stacked = tf.stack(scale_features, axis=-1)

        # Soft attention across scale axis
        attn_weights = tf.nn.softmax(
            tf.layers.conv2d(stacked, 1, [1, 1], activation=tf.nn.relu, name='scale_attn_conv'),
            axis=-1
        )  # [B, N, C, len(k_list)]

        fused = tf.reduce_sum(attn_weights * stacked, axis=-1)  # [B, N, C]

        # Optional local pooling
        locals_concat = tf.concat(local_features, axis=-1)  # [B, N, k, C]
        locals_max = tf.reduce_max(locals_concat, axis=-2, keepdims=False)  # [B, N, C]
        fused = tf.concat([fused, locals_max], axis=-1)  # [B, N, 2C]

        # Optional channel attention (SE block)
        #fused = channel_attention_block(fused, scope=scope + '_se')

        # Final projection + BN + ReLU
        fused = tf_util.conv1d(fused, 128, 1, padding='VALID', bn=True,
                               is_training=is_training, scope=scope + '_refine', bn_decay=bn_decay)

        # Dropout for regularization
        #fused = tf_util.dropout(fused, keep_prob=0.8, is_training=is_training, scope=scope + '_dp')

        return fused

    
def multi_scale_gap_block4(point_cloud, k_list, is_training, bn_decay, scope, is_dist=True):
    """
    Multi-scale GAPNet attention block (returns raw edge features and local features).
    
    Args:
        point_cloud: [B, N, 3]
        k_list: list of k-neighbor values (e.g., [10, 20, 30])
        Returns:
            multi_scale: [B, N, C_total] → concat of edge features from all scales
            locals_max:  [B, N, C_local] → max of local features across all scales
    """
    import tf_util
    from gat_layers3 import attn_feature  # GAPNet-style attention

    scale_features = []
    local_features = []
    n_heads=4

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for i, k in enumerate(k_list):
            adj_matrix = tf_util.pairwise_distance(point_cloud)        # [B, N, N]
            nn_idx = tf_util.knn(adj_matrix, k=k)                      # [B, N, k]

            layer_name = f"{scope}_attn_k{k}"
            with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                edge_feature, _, local = attn_feature(
                    point_cloud, 16, nn_idx,
                    activation=tf.nn.elu,
                    in_dropout=0.6, coef_dropout=0.6,
                    is_training=is_training, bn_decay=bn_decay,
                    layer=layer_name, k=k, i=0, is_dist=is_dist
                )
                scale_features.append(edge_feature)     # [B, N, 64]
                local_features.append(local)            # [B, N, k, 64]

        # Concatenate multi-scale edge features: [B, N, 64 * len(k_list)]
        multi_scale = tf.concat(scale_features, axis=-1)
        print(multi_scale.shape)
        # Concatenate local edge features from all scales: [B, N, sum(k), 64]
        locals_concat = tf.concat(local_features, axis=-2)
        print(locals_concat.shape)

        # Apply max pooling across neighbor dimension → [B, N, 64]
        locals_max = tf.reduce_max(locals_concat, axis=-2)
        print(locals_max.shape)
        locals_max = tf.expand_dims(locals_max, axis=2)

        return multi_scale, locals_max
    

    
    
def multi_scale_gap_block5(point_cloud, k_list, is_training, bn_decay, scope, is_dist=True):
    """
    Multi-scale GAPNet attention block (returns raw edge features and local features).
    
    Args:
        point_cloud: [B, N, 3]
        k_list: list of k-neighbor values (e.g., [10, 20, 30])
        Returns:
            multi_scale: [B, N, C_total] → concat of edge features from all scales
            locals_max:  [B, N, C_local] → max of local features across all scales
    """
    import tf_util
    from gat_layers3 import attn_feature  # GAPNet-style attention

    scale_features = []
    local_features = []

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for i, k in enumerate(k_list):
            adj_matrix = tf_util.pairwise_distance(point_cloud)        # [B, N, N]
            nn_idx = tf_util.knn(adj_matrix, k=k)                      # [B, N, k]

            layer_name = f"{scope}_attn_k{k}"
            with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                edge_feature, _, local = attn_feature(
                    point_cloud, 128, nn_idx,
                    activation=tf.nn.elu,
                    in_dropout=0.6, coef_dropout=0.6,
                    is_training=is_training, bn_decay=bn_decay,
                    layer=layer_name, k=k, i=0, is_dist=is_dist
                )
                scale_features.append(edge_feature)     # [B, N, 64]
                local_features.append(local)            # [B, N, k, 64]

        # Concatenate multi-scale edge features: [B, N, 64 * len(k_list)]
        multi_scale = tf.concat(scale_features, axis=-1)
        print(multi_scale.shape)
        # Concatenate local edge features from all scales: [B, N, sum(k), 64]
        locals_concat = tf.concat(local_features, axis=-2)
        print(locals_concat.shape)

        # Apply max pooling across neighbor dimension → [B, N, 64]
        locals_max = tf.reduce_max(locals_concat, axis=-2)
        print(locals_max.shape)
        locals_max = tf.expand_dims(locals_max, axis=2)

        return multi_scale, locals_max


def multi_scale_gap_block4_1(point_cloud, k_list, is_training, bn_decay, scope, is_dist=True):
    """
    Multi-scale, multi-head GAPNet attention block.
    Returns raw edge features and local features from each scale and head.
    
    Args:
        point_cloud: [B, N, C]
        k_list: list of k-neighbor values (e.g., [10, 20])
    
    Returns:
        multi_scale: [B, N, total_C]  → concat of edge features across scales and heads
        locals_max:  [B, N, total_C]  → max of local edge features across scales and heads
    """
    import tf_util
    from gat_layers3 import attn_feature

    scale_features = []
    local_features = []
    n_heads = 2  # number of attention heads

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for i, k in enumerate(k_list):
            adj_matrix = tf_util.pairwise_distance(point_cloud)   # [B, N, N]
            nn_idx = tf_util.knn(adj_matrix, k=k)                 # [B, N, k]

            head_outputs = []
            head_locals = []

            for h in range(n_heads):
                head_scope = f"{scope}_attn_k{k}_head{h}"
                with tf.variable_scope(head_scope, reuse=tf.AUTO_REUSE):
                    edge_feature, _, local = attn_feature(
                        point_cloud, 64, nn_idx,
                        activation=tf.nn.elu,
                        in_dropout=0.6, coef_dropout=0.6,
                        is_training=is_training, bn_decay=bn_decay,
                        layer=head_scope, k=k, i=h, is_dist=is_dist
                    )
                    print('edge',edge_feature.shape,local.shape)
                    head_outputs.append(edge_feature)   # [B, N, 16]
                    head_locals.append(local)           # [B, N, k, 16]

            # Concatenate heads for this scale
            scale_feature = tf.concat(head_outputs, axis=-1)      # [B, N, 16*n_heads]
            local_feature = tf.concat(head_locals, axis=-1)       # [B, N, k, 16*n_heads]
            print('scale',scale_feature.shape,local_feature.shape)
            scale_features.append(scale_feature)
            local_features.append(local_feature)
            

        # Concatenate all scales
        multi_scale = tf.concat(scale_features, axis=-1)          # [B, N, len(k_list)*16*n_heads]
        locals_concat = tf.concat(local_features, axis=-2)        # [B, N, sum(k), 16*n_heads]
        locals_max = tf.reduce_max(locals_concat, axis=-2)        # [B, N, 16*n_heads]
        locals_max = tf.expand_dims(locals_max, axis=2)

        return multi_scale, locals_max


    
    
def dynamic_graph_edgeconv(inputs, k, out_dim, is_training, bn_decay, scope):
    """
    DGCNN-style dynamic edge convolution with batch-safe reshape.
    inputs: [B, N, C]
    returns: [B, N, out_dim]
    """
    import tf_util
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        adj_matrix = tf_util.pairwise_distance(inputs)
        nn_idx = tf_util.knn(adj_matrix, k=k)

        # Neighbors and central points
        neighbors = tf_util.get_neighbors(inputs, nn_idx, k=k)           # [B, N, k, C]
        central = tf.gather(inputs, nn_idx, batch_dims=1)                # [B, N, k, C]

        # Safe reshaping to avoid squeeze issue
        central = tf.reshape(central, tf.shape(neighbors))               # [B, N, k, C]

        # Edge features: central + relative difference
        edge_feature = tf.concat([central, neighbors - central], axis=-1)  # [B, N, k, 2C]

        net = tf_util.conv2d(edge_feature, out_dim, [1, 1],
                             bn=True, is_training=is_training,
                             scope='dgcnn_conv', bn_decay=bn_decay)
        net = tf.reduce_max(net, axis=2)  # Aggregate over k neighbors → [B, N, out_dim]
        return net


def gcn_reasoning_block(inputs, k, is_training, bn_decay, scope, is_dist=True):
    """
    Lightweight GCN-style block for reasoning over point features.
    Args:
        inputs: [B, N, C]
        k: number of neighbors
    Returns:
        [B, N, C_out] - message-passed features
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        adj_matrix = tf_util.pairwise_distance(inputs)
        nn_idx = tf_util.knn(adj_matrix, k=k)

        neighbors = tf_util.get_neighbors(inputs, nn_idx=nn_idx, k=k)  # [B, N, k, C]
        central = tf.expand_dims(inputs, axis=2)                       # [B, N, 1, C]
        central_tiled = tf.tile(central, [1, 1, k, 1])                 # [B, N, k, C]

        message = tf.concat([central_tiled, neighbors], axis=-1)      # [B, N, k, 2C]
        message = tf_util.conv2d(message, 128, [1, 1], padding='VALID', stride=[1, 1],
                                 bn=True, is_training=is_training, scope='gcn_mlp',
                                 bn_decay=bn_decay, is_dist=is_dist)

        message_pooled = tf.reduce_max(message, axis=2)               # [B, N, 128]

        return message_pooled


def knn_point1(k, xyz1, xyz2):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    b = xyz1.get_shape()[0].value
    n = xyz1.get_shape()[1].value
    c = xyz1.get_shape()[2].value
    m = xyz2.get_shape()[1].value
    print(b, n, c, m)
    print(xyz1, (b,1,n,c))
    xyz1 = tf.tile(tf.reshape(xyz1, (b,1,n,c)), [1,m,1,1])
    xyz2 = tf.tile(tf.reshape(xyz2, (b,m,1,c)), [1,1,n,1])
    dist = tf.reduce_sum((xyz1-xyz2)**2, -1)
    print(dist, k)
    outi, out = tf.nn.top_k(k, dist)
    idx = tf.slice(outi, [0,0,0], [-1,-1,k])
    val = tf.slice(out, [0,0,0], [-1,-1,k])
    print(idx, val)
    #val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return val, idx

def knn_point(k, xyz1, xyz2):
    '''
    Input:
        k: int32
        xyz1: (B, N, C) float32 tensor — input points
        xyz2: (B, M, C) float32 tensor — query points
    Output:
        val: (B, M, k) float32 — L2 distances
        idx: (B, M, k) int32 — indices of k nearest neighbors in xyz1
    '''
    b = xyz1.get_shape()[0].value
    n = xyz1.get_shape()[1].value
    c = xyz1.get_shape()[2].value
    m = xyz2.get_shape()[1].value

    xyz1_tiled = tf.tile(tf.reshape(xyz1, (b, 1, n, c)), [1, m, 1, 1])
    xyz2_tiled = tf.tile(tf.reshape(xyz2, (b, m, 1, c)), [1, 1, n, 1])
    
    dist = tf.reduce_sum((xyz1_tiled - xyz2_tiled) ** 2, axis=-1)  # [B, M, N]

    val, idx = tf.nn.top_k(-dist, k=k)  # returns k smallest distances
    return -val, idx


def group_point(points, idx):
    '''
    Group features from input points using KNN indices.
    
    Input:
        points: (B, N, C) tensor
        idx: (B, M, K) tensor — indices of neighbors

    Output:
        grouped_points: (B, M, K, C)
    '''
    B = tf.shape(points)[0]
    N = tf.shape(points)[1]
    C = tf.shape(points)[2]
    M = tf.shape(idx)[1]
    K = tf.shape(idx)[2]

    batch_indices = tf.range(B)[:, tf.newaxis, tf.newaxis]  # [B, 1, 1]
    batch_indices = tf.tile(batch_indices, [1, M, K])       # [B, M, K]

    idx_expanded = tf.stack([batch_indices, idx], axis=-1)  # [B, M, K, 2]
    grouped_points = tf.gather_nd(points, idx_expanded)     # [B, M, K, C]
    
    return grouped_points


def sample_and_group(nsample, xyz, points, knn=True, use_xyz=True):
    """
    Sample all points and group k neighbors for each.
    Args:
        nsample: int
        xyz: [B, N, 3]
        points: [B, N, C]
    Returns:
        grouped_xyz: [B, N, nsample, 3]
        new_points: [B, N, nsample, 3+C] if use_xyz else [B, N, nsample, C]
        idx: [B, N, nsample]
    """
    _, idx = knn_point(nsample, xyz, xyz)  # [B, N, nsample]
    grouped_xyz = group_point(xyz, idx)    # [B, N, nsample, 3]
    grouped_xyz -= tf.expand_dims(xyz, 2)  # [B, N, nsample, 3]

    if points is not None:
        grouped_points = group_point(points, idx)  # [B, N, nsample, C]
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return grouped_xyz, new_points, idx


def point_transformer_block(xyz, features, is_training, bn_decay, scope, k=16, C=64):
    '''
    xyz:      (B, N, 3)       - input point coordinates
    features: (B, N, C_in)    - input features
    Return:   (B, N, C)       - output features
    '''
    with tf.variable_scope(scope):
        # 1. Compute pairwise distances and get k-NN indices
        adj_matrix = tf_util.pairwise_distance(xyz)
        nn_idx = tf_util.knn(adj_matrix, k=k)

        # 2. Group coordinates and features
        grouped_xyz = group_point(xyz, nn_idx)  # [B, N, k, 3]
        xyz_tile = tf.expand_dims(xyz, axis=2)  # [B, N, 1, 3]
        pos_enc = grouped_xyz - xyz_tile        # [B, N, k, 3] — relative positional encoding

        # 3. Group features
        grouped_feat = group_point(features, nn_idx)  # [B, N, k, C_in]

        # 4. Positional Encoding → project to same dimension as features
        pos_enc = tf_util.conv2d(pos_enc, C, [1, 1], bn=True, is_training=is_training,
                                 scope=scope + '_pos_enc2', bn_decay=bn_decay)  # [B, N, k, C]

        # 5. Linear projection of grouped features
        phi = tf_util.conv2d(grouped_feat, C, [1, 1], bn=True, is_training=is_training,
                             scope=scope + '_phi', bn_decay=bn_decay)  # [B, N, k, C]

        # 6. Attention weights
        attn_feat = tf_util.conv2d(grouped_feat, C, [1, 1], bn=True, is_training=is_training,
                                   scope=scope + '_attn', bn_decay=bn_decay)  # [B, N, k, C]
        attn_weights = tf.nn.softmax(attn_feat, axis=2)  # softmax over k neighbors

        # 7. Weighted sum
        out = tf.reduce_sum(attn_weights * (phi + pos_enc), axis=2)  # [B, N, C]

       
        
        
        return out



def attn_feature_enhanced(input_feature, output_dim, neighbors_idx, activation=tf.nn.elu,
                          in_dropout=0.1, coef_dropout=0.1, is_training=True, bn_decay=None,
                          layer='', k=20, i=0, is_dist=False):
    """
    Enhanced single-head attention feature aggregation:
    - Richer edge features: [central, neighbor, difference, distance]
    - Gating + residual connections
    - Dropout for regularization
    """
    # Shape: input_feature [B, N, C]
    input_feature = tf.expand_dims(input_feature, axis=2)  # [B, N, 1, C]
    neighbors = tf_util.get_neighbors(input_feature, nn_idx=neighbors_idx, k=k)  # [B, N, k, C]
    
    # Tiling for edge computation
    input_tiled = tf.tile(input_feature, [1, 1, k, 1])  # [B, N, k, C]
    
    # Distance encoding
    distance = tf.norm(input_tiled - neighbors, axis=-1, keepdims=True)  # [B, N, k, 1]
    
    # Rich edge features: concat(central, neighbor, difference, distance)
    edge_feature = tf.concat([input_tiled, neighbors, input_tiled - neighbors, distance], axis=-1)  # [B, N, k, 3C+1]

    # MLP on edge features
    edge_proj = tf_util.conv2d(edge_feature, output_dim, [1, 1], padding='VALID', stride=[1, 1],
                               bn=True, is_training=is_training, scope=layer+'_edge_proj_'+str(i),
                               bn_decay=bn_decay, is_dist=is_dist)

    # Attention coefficients from edge features
    logits = tf_util.conv2d(edge_proj, 1, [1, 1], padding='VALID', stride=[1, 1],
                            bn=False, activation_fn=None, is_training=is_training,
                            scope=layer+'_att_logits_'+str(i), is_dist=is_dist)
    logits = tf.transpose(logits, [0, 1, 3, 2])  # [B, N, 1, k] -> [B, N, k, 1]
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits), axis=2)

    # Dropout for regularization
    if coef_dropout > 0.0:
        coefs = tf.nn.dropout(coefs, keep_prob=1.0 - coef_dropout)

    # Weighted sum of edge features
    aggregated = tf.matmul(coefs, edge_proj)  # [B, N, 1, C]

    # Residual connection: add transformed central feature
    central_proj = tf_util.conv2d(input_feature, output_dim, [1, 1], bn=True,
                                  is_training=is_training, scope=layer+'_residual_'+str(i),
                                  bn_decay=bn_decay)
    ret = activation(aggregated + central_proj)
    ret = tf.squeeze(ret, axis=2)  # [B, N, C]
    
    return ret, coefs, edge_feature
