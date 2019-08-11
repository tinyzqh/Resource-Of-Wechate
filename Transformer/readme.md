## 传统递归网络的问题

&emsp;&emsp;处理Seq2seq最常用的就是RNN。RNN的问题在于无法Parallel(并行处理)，可以用CNN解决这个问题，但是CNN能够考虑的特征向量非常少，而解决这个问题又可以通过再次叠加CNN来解决。这样的话CNN就可以解决RNN不能处理Parallel的问题。

![RNN和CNN并行网络结构图](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/1.png)

&emsp;&emsp;但是也存在问题，就是CNN需要叠加很多层。为了解决这个问题，又引入Self-Attention Layer，其输入是一个sequence输出也是一个sequence，能够达到跟RNN一样的功效，输出`b`可以平行同时计算出来。


![Self-Attention示意图](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/2.png)

## 解决办法 Attention Is All You Need

&emsp;&emsp;其技术最早出现的那篇文章就是[Attention Is All You Need](https://arxiv.org/abs/1706.03762) 。输入首先经过一个embedding输出<a href="https://www.codecogs.com/eqnedit.php?latex=a_1-a_4" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a_1-a_4" title="a_1-a_4" /></a>。输出再乘上三个Matrix：`q`、`k`、`v`。`q`用于match其它输出，`k`用于被match，`v`是抽取出来的信息。

![q、k、v生成示意图](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/3.png)

&emsp;&emsp;之后拿每一个query `q`去对每个key `k`做attention。Attention是吃两个向量，输出这两个向量有多匹配(输出一个分数)，做Attention的方法有很多种。

![Scaled Dot-Product Attention示意图](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/4.png)

&emsp;&emsp;除以$\sqrt{d}$是相当于归一化的处理。之后经过softmax得到$\hat{\alpha}$：

![归一化](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/5.png)

&emsp;&emsp;再将$\hat{\alpha}$与`v`相乘得到`b`：

![得到最终结果](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/6.png)

&emsp;&emsp;如果self-attention的$\hat{\alpha}$等于0，那么他就会得到local的attention。也就是说attention可以决定看哪些需要的信息。也可以同时用$q^2$做attention之后计算$b^2$。

![计算下一个](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/7.png)

&emsp;&emsp;从下面矩阵相称的方法里面可以更清楚地看到并行化处理的方式：

![矩阵并行处理-1](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/8.png)

![矩阵并行处理-2](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/9.png)

&emsp;&emsp;更进一步得到：

![矩阵并行处理-3](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/10.png)

![矩阵并行处理-4](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/11.png)

&emsp;&emsp;整个计算流程可以使用下图表示：

![矩阵并行处理-5](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/12.png)

&emsp;&emsp;输入乘以matrix得到query、key和抽取的信息`v`。`K`与`Q`相乘得到Attention后做softmax得到$\hat{a}$，再与`V`相乘得到输出。
 
&emsp;&emsp;Self-attention有个变形Multi-head。举一个两个head的例子：

![Multi-head Self-attention-1](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/13.png)

&emsp;&emsp;$q^{i,1}$与$k^{i,1}$、$k^{j,1}$相乘得到$b^{i,1}$，$q^{i,2}$与$k^{i,2}$、$k^{j,2}$相乘得到$b^{i,2}$。再将两者接起来得到self-attention的输出：

![Multi-head Self-attention-2](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/14.png)

&emsp;&emsp;Multi-head Self-attention所期望的就是不同的head能够关注不同的东西，比如有的head关注局部信息，而另外一些关注全局信息。

&emsp;&emsp;对self-attention来说，input的次序是不重要的。这样的话就会导致一个问题，比如说语句“A打了B”跟“B打了A”是一样的。但是我们希望将Input的次序考虑进去。在原始的论文中，作者加入设定的$e^i$(不是学习出来的)来解决这个问题。相当于提供位置资讯。

![位置编码](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/15.png)

## Transformer使用

&emsp;&emsp;上面讲的是self-attention取代RNN。接下来我们阐述self-attention怎么在seq2seq中使用：

![Sequence to Sequence](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/16.png)

&emsp;&emsp;简要来说就是Encoder中的RNN与Decoder中的RNN统统都用Self-Attention来代替。

&emsp;&emsp;Transform的网络结构如下所示：

![Transformer网络结构示意图](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/17.png)

&emsp;&emsp;数据先加上位置关系进入编码器，在编码器中先经过Self-Attention Layer 再通过add和layer norm层。再Decode中输入是前一个time step的output，经过编码和位置编码之后输入到block中，Masked Multi-Head中的Mask用于处理之前发生过的状态做attention。再与之前的Input Embedding得到的输出结合，得到最终的输出。

&emsp;&emsp;更进一步有[Universal Transformer](https://ai.googleblog.com/2018/08/moving-beyond-translation-with.html)。

![Universal Transformer](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/18.png)

&emsp;&emsp;最早Transform用于文字上，现在也可被用于影像上面。[Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)

![Transformer用于影像](https://github.com/ZhiqiangHo/Resource-Of-Wechate/blob/master/Transformer/figure/19.png)

## 代码解释

&emsp;&emsp;在开始之前要先下载数据集：这里附上百度云盘链接：https://pan.baidu.com/s/1J6p8wo5xyymhLhoSyP_X4Q 密码：wzrn。完整版[代码连接](https://github.com/ZhiqiangHo/Resource-Of-Wechate/tree/master/Transformer)。

- 介绍`transform.py`这个文件：

&emsp;&emsp;在`transform.py`里面有两个类`params`和`Transform`。在中主要配置一些参数:

```python
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
```

&emsp;&emsp;在`Transform`中主要介绍了一些函数：

&emsp;&emsp;`layer normalization`层，如果不是很懂的话就理解为归一化层。

```python
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
```

&emsp;&emsp;`scaled_dot_product_attention`注意力机制层：
&emsp;&emsp;其公式是$\alpha_{1,i}=q^{i}·k^{i}/ \sqrt{d}$，我们下面这个函数就是要实现这个功能。拿到数据之后先获取`d`的维度`d_k`，之后通过`tf.transpose()`函数将数据的第二维变为第一维，第一维度变为第二维度，为与`Q`相乘的矩阵运算做准备。得到最终结果之后除以`d`的开方，这样会对`softmax`反向传播有利。之后再经过`softmax`与`V`相乘。
```python
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
```

&emsp;&emsp;`Multi-head attention` 将 `Q`、 `K`、 `V` 通过一个线性映射之后，分 成`h`份，对每一份进行`scaled dot-product attention` 效果更好。 然后,把各个部分的结果合并起来，再次经过线性映射，得到最终的输出。

```python
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
```

&emsp;&emsp;`Residual connection`将网络进行残差连接，有利于反向传播：

```python
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
```

&emsp;&emsp;下面是按照逻辑，将整个网络连接起来：

```python
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
```

- 介绍`mnist_transformer.py`这个文件：

&emsp;&emsp;这里主要是主程序调用`Transformer`函数实现mnist识别主程序：

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data
from mnist_transformers.transformer import Transform
tf.set_random_seed(1)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
lr = 0.001
training_iters = 100000
batch_size = 128
n_input = 28 #列 输入的图片数据为28*28
# 时序持续长度为28，即每做一次预测，需要先输入28行
n_step = 28 #行
n_classes = 10

x = tf.placeholder(tf.float32, [None,n_step,n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

pred = Transform().transform(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
Saver = tf.train.Saver()

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # try:
    #     Saver.restore(sess, tf.train.latest_checkpoint("E://code_of_ocr/code_of_rnn/code_of_rnn_mnist/network_model"))
    #     print('success add the model')
    # except:
    #     sess.run(tf.global_variables_initializer())
    #     print('error of add the model')
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_step, n_input])
        sess.run([train_op], feed_dict={x: batch_xs,y: batch_ys})
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x: batch_xs,y: batch_ys}))
        step += 1
    # Saver.save(sess, "E://code_of_ocr/code_of_rnn/code_of_rnn_mnist/network_model/crack_capcha.model")
```


## 参考链接

- [李宏毅深度学习](https://www.bilibili.com/video/av46561029/?p=60)
