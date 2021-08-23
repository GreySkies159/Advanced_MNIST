import tensorflow._api.v2.compat.v1 as tf
# tf.gfile = tf.io.gfile
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

tf.disable_v2_behavior()

# Saving the graph as a pb file taking data from pbtxt and ckpt files and providing a few operations
freeze_graph.freeze_graph(input_graph='advanced_mnist.pbtxt',
                          input_saver='',
                          input_binary=True,
                          input_checkpoint='advanced_mnist.ckpt',
                          output_node_names='y_readout1',
                          restore_op_name='save/restore_all',
                          filename_tensor_name='save/Const:0',
                          output_graph='frozen_advanced_mnist.pb',
                          clear_devices=True,
                          initializer_nodes='')

# Read the data form the frozen graph pb file
input_graph_def = tf.GraphDef()
with tf.gfile.Open('frozen_advanced_mnist.pb', 'rb') as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

# Optimize the graph with input and output nodes
output_graph_def = optimize_for_inference_lib.optimize_for_inference(
    input_graph_def,
    ['x_input', 'keep_prob'],
    ['y_readout1'],
    tf.float32.as_datatype_enum)

# Save the optimized graph to the optimized pb file
f = tf.gfile.FastGFile('optimized_advanced_mnist.pb', 'w')
f.write(output_graph_def.SerializeToString())
