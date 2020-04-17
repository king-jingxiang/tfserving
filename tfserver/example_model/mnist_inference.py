import tensorflow as tf
import numpy as np

saved_model_dir = "/Users/jinxiang/Downloads/models/2/1"

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], saved_model_dir)
    input_operator = sess.graph.get_tensor_by_name("ParseExample/ParseExample:0")
    output_operator = sess.graph.get_tensor_by_name("y:0")
    inputs = np.random.normal(0, 1, [1,784])
    results = sess.run(output_operator, feed_dict={input_operator: inputs})
    print(results)
