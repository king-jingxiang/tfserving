import tensorflow as tf
import numpy as np

saved_model_dir = "/Users/jinxiang/Downloads/models/2/1"


def inference():
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(sess, ["serve"], saved_model_dir)
        input_operator = sess.graph.get_tensor_by_name("ParseExample/ParseExample:0")
        output_operator = sess.graph.get_tensor_by_name("y:0")
        inputs = np.random.normal(0, 1, [1, 784])
        results = sess.run(output_operator, feed_dict={input_operator: inputs})
        print(results)


def inference2():
    # graph = tf.Graph()
    sess = tf.Session(graph=tf.Graph())
    with sess.as_default():
        tf.saved_model.loader.load(sess, ["serve"], saved_model_dir)
    input_operator = sess.graph.get_tensor_by_name("ParseExample/ParseExample:0")
    output_operator = sess.graph.get_tensor_by_name("y:0")
    inputs = np.random.normal(0, 1, [1, 784])
    with sess.as_default():
        results = sess.run(output_operator, feed_dict={input_operator: inputs})
        print(results)


def inference3():
    graph = tf.compat.v1.Graph()
    with tf.compat.v1.Session(graph=graph) as sess:
        with graph.as_default():
            tf.compat.v1.saved_model.loader.load(sess, ["serve"], saved_model_dir)
    input_operator = graph.get_tensor_by_name("ParseExample/ParseExample:0")
    output_operator = graph.get_tensor_by_name("y:0")
    inputs = np.random.normal(0, 1, [1, 784])

    with tf.compat.v1.Session(graph=graph) as sess:
        results = sess.run(output_operator, feed_dict={input_operator: inputs})
        print(results)


if __name__ == '__main__':
    # inference()
    inference2()
    # inference3()
