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