import sys
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.saved_model import tag_constants
from tensorflow.core.protobuf import saver_pb2

PATH = r'hellotensor_2/'
MODEL_NAME = 'saved_model'

# Freeze the graph

input_graph_path = PATH + MODEL_NAME + '.pb'
checkpoint_path = PATH + 'variables/variables'
input_saver_def_path = ""
input_binary = True
output_node_names = "O"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name =  PATH + 'frozen_'+MODEL_NAME+'.pb'
output_optimized_graph_name = PATH + 'optimized_'+MODEL_NAME+'.pb'
clear_devices = True

freeze_graph.freeze_graph(input_graph = input_graph_path,
                 input_saver = input_saver_def_path,
                 input_binary = input_binary,
                 input_checkpoint = checkpoint_path,
                 output_node_names = output_node_names,
                 restore_op_name = restore_op_name,
                 filename_tensor_name = filename_tensor_name,
                 output_graph = output_frozen_graph_name,
                 clear_devices = clear_devices,
                 initializer_nodes= "",
                 input_meta_graph=None,
                 input_saved_model_dir=PATH,
                 saved_model_tags=tag_constants.SERVING,
                 checkpoint_version=saver_pb2.SaverDef.V2)
