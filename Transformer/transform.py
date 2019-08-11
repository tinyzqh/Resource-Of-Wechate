# -*- coding: utf-8 -*-
# /usr/bin/python3
import numpy as np
import tensorflow as tf
import argparse
class params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_size', default=28, type=int)
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
    hparams = params()
    parser = hparams.parser
    hp = parser.parse_args()
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

            # softmax
            outputs = tf.nn.softmax(outputs)

            # dropout
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

            # weighted sum (context vectors)
            outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

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
            enc = tf.layers.dense(xs, self.hp.d_model, activation=None)
            enc = tf.reshape(enc, shape=[-1, 28, self.hp.d_model])
            enc *= self.hp.d_model**0.5 # scale

            # enc += positional_encoding(enc, self.hp.maxlen1)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)
            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = self.multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = self.ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        flatten = tf.layers.flatten(enc)
        logist = tf.layers.dense(flatten, 10, activation=tf.nn.sigmoid)
        return logist
if __name__ == "__main__":

    import tensorflow as tf
    data = np.ones((2, 28, 28))
    inputs = tf.convert_to_tensor(data, tf.float32)
    p = Transform().transform(inputs)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run([p]))

