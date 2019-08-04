# -*- coding: utf-8 -*-
# /usr/bin/python3


import numpy as np
import argparse
class params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_size', default=29, type=int)
    parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--num_blocks', default=1, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
class Transform():
    def ln(self, inputs, epsilon=1e-8, scope="ln"):
        '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
        inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
        epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        scope: Optional scope for `variable_scope`.

        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def scaled_dot_product_attention(self, Q, K, V,
                                     causality=False, dropout_rate=0.,
                                     training=True,
                                     scope="scaled_dot_product_attention"):
        '''See 3.2.1.
        Q: Packed queries. 3d tensor. [N, T_q, d_k].
        K: Packed keys. 3d tensor. [N, T_k, d_k].
        V: Packed values. 3d tensor. [N, T_k, d_v].
        causality: If True, applies masking for future blinding
        dropout_rate: A floating point number of [0, 1].
        training: boolean for controlling droput
        scope: Optional scope for `variable_scope`.
        '''
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            d_k = Q.get_shape().as_list()[-1]

            # dot product
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

            # scale
            outputs /= d_k ** 0.5

            # key masking
            outputs = self.mask(outputs, Q, K, type="key")

            # causality or future blinding masking
            if causality:
                outputs = self.mask(outputs, type="future")

            # softmax
            outputs = tf.nn.softmax(outputs)
            attention = tf.transpose(outputs, [0, 2, 1])
            tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

            # query masking
            outputs = self.mask(outputs, Q, K, type="query")

            # dropout
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

            # weighted sum (context vectors)
            outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

        return outputs

    def mask(self, inputs, queries=None, keys=None, type=None):
        """Masks paddings on keys or queries to inputs
        inputs: 3d tensor. (N, T_q, T_k)
        queries: 3d tensor. (N, T_q, d)
        keys: 3d tensor. (N, T_k, d)

        e.g.,
        >> queries = tf.constant([[[1.],
                            [2.],
                            [0.]]], tf.float32) # (1, 3, 1)
        >> keys = tf.constant([[[4.],
                         [0.]]], tf.float32)  # (1, 2, 1)
        >> inputs = tf.constant([[[4., 0.],
                                   [8., 0.],
                                   [0., 0.]]], tf.float32)
        >> mask(inputs, queries, keys, "key")
        array([[[ 4.0000000e+00, -4.2949673e+09],
            [ 8.0000000e+00, -4.2949673e+09],
            [ 0.0000000e+00, -4.2949673e+09]]], dtype=float32)
        >> inputs = tf.constant([[[1., 0.],
                                 [1., 0.],
                                  [1., 0.]]], tf.float32)
        >> mask(inputs, queries, keys, "query")
        array([[[1., 0.],
            [1., 0.],
            [0., 0.]]], dtype=float32)
        """
        padding_num = -2 ** 32 + 1
        if type in ("k", "key", "keys"):
            # Generate masks
            masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
            masks = tf.expand_dims(masks, 1)  # (N, 1, T_k)
            masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)

            # Apply masks to inputs
            paddings = tf.ones_like(inputs) * padding_num
            outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)
        elif type in ("q", "query", "queries"):
            # Generate masks
            masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
            masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
            masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)

            # Apply masks to inputs
            outputs = inputs * masks
        elif type in ("f", "future", "right"):
            diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

            paddings = tf.ones_like(masks) * padding_num
            outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
        else:
            print("Check if you entered type correctly!")

        return outputs


    def multihead_attention(self, queries, keys, values,
                            num_heads=8,
                            dropout_rate=0,
                            training=True,
                            causality=False,
                            scope="multihead_attention"):
        '''Applies multihead attention. See 3.2.2
        queries: A 3d tensor with shape of [N, T_q, d_model].
        keys: A 3d tensor with shape of [N, T_k, d_model].
        values: A 3d tensor with shape of [N, T_k, d_model].
        num_heads: An int. Number of heads.
        dropout_rate: A floating point number.
        training: Boolean. Controller of mechanism for dropout.
        causality: Boolean. If true, units that reference the future are masked.
        scope: Optional scope for `variable_scope`.

        Returns
          A 3d tensor with shape of (N, T_q, C)
        '''
        d_model = queries.get_shape().as_list()[-1]
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Linear projections
            Q = tf.layers.dense(queries, d_model, use_bias=False)  # (N, T_q, d_model)
            K = tf.layers.dense(keys, d_model, use_bias=False)  # (N, T_k, d_model)
            V = tf.layers.dense(values, d_model, use_bias=False)  # (N, T_k, d_model)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

            # Attention
            outputs = self.scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

            # Residual connection
            outputs += queries

            # Normalize
            outputs = self.ln(outputs)

        return outputs

    def ff(self, inputs, num_units, scope="positionwise_feedforward"):
        '''position-wise feed forward net. See 3.3

        inputs: A 3d tensor with shape of [N, T, C].
        num_units: A list of two integers.
        scope: Optional scope for `variable_scope`.

        Returns:
          A 3d tensor with the same shape and dtype as inputs
        '''
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Inner layer
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

            # Outer layer
            outputs = tf.layers.dense(outputs, num_units[1])

            # Residual connection
            outputs += inputs

            # Normalize
            outputs = self.ln(outputs)

        return outputs
    def transform(self, xs, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            # x, seqlens, sents1 = xs

            # embedding
            # enc = tf.nn.embedding_lookup(self_embeddings, xs) # (N, T1, d_model)
            enc = tf.layers.dense(xs, hp.d_model, activation=None)
            enc = tf.reshape(enc, shape=[-1, 10, hp.d_model])
            enc *= hp.d_model**0.5 # scale

            # enc += positional_encoding(enc, self.hp.maxlen1)
            enc = tf.layers.dropout(enc, hp.dropout_rate, training=training)
            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = self.multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=hp.num_heads,
                                              dropout_rate=hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = self.ff(enc, num_units=[hp.d_ff, hp.d_model])
        flatten = tf.layers.flatten(enc)
        logist = tf.layers.dense(flatten, 1, activation=None)
        return logist
if __name__ == "__main__":

    hparams = params()
    parser = hparams.parser
    hp = parser.parse_args()

    import tensorflow as tf
    data = np.ones((2, 10, 29))
    inputs = tf.convert_to_tensor(data, tf.float32)
    p = Transform().transform(inputs)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run([p]))





