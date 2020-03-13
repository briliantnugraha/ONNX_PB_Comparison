import onnxruntime as rt
import numpy as np
import cv2
import os
import onnx
import tensorflow as tf
from onnx import helper
from onnx import TensorProto

#basic path
path = r'D:\forYaoying' # put your path here
pathmodel = os.path.join(path,'frozen.onnx') # put your frozen onnx model here


# put some image input
img = cv2.imread(os.path.join(path,'testimage.jpg')).astype(np.uint8) # put your image input here
#img = cv2.resize(img, (1152, 800))
img_preprocess = np.expand_dims(img, 0) #/ 255)#.astype('uint8')
#img_preprocess = np.moveaxis(img_preprocess, [1,2,3], [2,3,1])


#######################################################################
'''
ref: https://github.com/microsoft/onnxruntime/issues/1455#issuecomment-514805365
This is ONNX debug
The purpose is to inspect the desired intermediate layer from onnx

Flow:
    - A. Load the model and do some sanity check on it
    - A*. You could inspect the input/output here
    - B. Get the model node (length and node names in string)
        *Add this step, you could use any editor, and CTRL+F your desired layer
        in my case, I want to get Resnet101 with block4 output
    - C. Put the name into some variables
    - D. Add the intermediate layers' name into the helper function
    - E. Add the helper function into model output
    - F. Save the model with the added outputs
    - G. Load the customized model
    - H. Inspect the new I/O's names of the customized model here
    - I. Do some inference on your image, and get/save the output
    
'''

# Step A
# load and check that the model is correct or not
model = onnx.load(pathmodel)
onnx.checker.check_model(model)

# Step A*
#sess = rt.InferenceSession(os.path.join(path,'frozen.onnx'))
##sess = rt.InferenceSession('D:\\embedmask_simplifier.onnx')
#input1 = sess.get_inputs()[0].name
#outputs = sess.run([], {input1: img_preprocess})

# Step B
# get the output of block 4 add and relu operation for each add layer
node_count = len(model.graph.node)
model_node = str(model.graph.node)

# Step C
# find the name first
unit1_relu_name = 'SecondStageFeatureExtractor/resnet_v1_101/block4/unit_1/bottleneck_v1/Relu:0'
unit2_relu_name = 'SecondStageFeatureExtractor/resnet_v1_101/block4/unit_2/bottleneck_v1/Relu:0'
unit3_relu_name = 'SecondStageFeatureExtractor/resnet_v1_101/block4/unit_3/bottleneck_v1/Relu:0'

# Step D
# add into the current onnx model and save to a new model
info_unit1_relu_name = helper.ValueInfoProto()
info_unit1_relu_name.name = unit1_relu_name
info_unit2_relu_name = helper.ValueInfoProto()
info_unit2_relu_name.name = unit2_relu_name
info_unit3_relu_name = helper.ValueInfoProto()
info_unit3_relu_name.name = unit3_relu_name

# Step E
model.graph.output.extend([info_unit1_relu_name, info_unit2_relu_name, info_unit3_relu_name])

# Step F
onnx.save(model, os.path.join(path,'frozen_out.onnx'))
# Step G
# load the modified model with the intermediate output that you want to check
sess = rt.InferenceSession(os.path.join(path,'frozen_out.onnx'))

# Step H
outputs = sess.get_outputs()
# check the output name
for out in outputs:
    print(out.name, out.shape)
    
# Step I
# check the output
input1_modify = sess.get_inputs()[0].name
outputs_modify = sess.run([], {input1_modify: img_preprocess})
np.save(os.path.join(path,'onnx_output'), outputs_modify)


#######################################################################
'''
This is PB debug
Thanks to Pawn in helping me to provide the code to get frozen graph(ops) names

Flow:
    - A. Load frozen graph of TF
    - B. read graph
    - C. Get the graph/ops operations
    - D. Get the graph/ops names
    - E. Find the graphs/ops names that match with our desired intermediate layers
    - F. Add the tensor_name (instead of just str name*) to a list
    - G. Do inference (sess.run) to with our list
    - H. Save the inference output to .npy
'''

# Step A
print('Load frozen graph of Tensorflow...')
PATH_TO_CKPT = os.path.join(path,'frozen_inference_graph.pb')
detection_graph = tf.Graph()
with detection_graph.as_default():
    # Step B
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')        
        # Step C
        ops = detection_graph.get_operations()
        # Step D
        ops_name = [o.name for o in ops]
        
    # Step E
    print('get output by name: ', [unit1_relu_name, unit2_relu_name, unit3_relu_name])
    get_ops = []
    for o in ops_name:
        if 'SecondStageFeatureExtractor/resnet_v1_101/block4/unit_1/bottleneck_v1/Relu' in o or \
            'SecondStageFeatureExtractor/resnet_v1_101/block4/unit_2/bottleneck_v1/Relu' in o or \
            'SecondStageFeatureExtractor/resnet_v1_101/block4/unit_3/bottleneck_v1/Relu' in o:
            print('output the same: ', o)
        else:
            continue
        # Step F
        get_ops.append(detection_graph.get_tensor_by_name(o+':0'))

# Step G
print('Do sess.run to get the output of intermediate layers...')
with tf.Session(graph=detection_graph) as sess:
    tf_outputs = sess.run(get_ops, feed_dict={'image_tensor:0': img_preprocess})
    
# Step H
np.save(os.path.join(path,'tf_output'), tf_outputs)


#######################################################################
'''
This is ONNX and PB intermediate output comparison

NOTE:
    This step depends on the previous .npy outputs
    The ONNX of intermediate arrays that I saved will start from index 5

Flow:
    - A. Load both the ONNX and the PB npy files
    - B. For each intermediate layers
    - C. Get the intermediate layer of both ONNX and PB
    - D. Compare the intermediate layer of ONNX and PB
    - E. Get the sum of it and compare with the supposed sum from the shape of the array
'''

# Step A
print('compare ONNX and Tensorflow frozen graph intermediate layers')
onnx_output = list(np.load(os.path.join(path,'onnx_output.npy'), allow_pickle=True) )
tf_output = list( np.load(os.path.join(path,'tf_output.npy'), allow_pickle=True) )

# Step B
for i in range(len(tf_output)):
    # Step C
    # compare layer 1 relu
    compare_layer_onnx = np.moveaxis(onnx_output[i+5], [0,1,2,3], [0,3,1,2])
    compare_layer_tf = tf_output[0]
    # Step E
    compare_output = compare_layer_onnx == compare_layer_tf
    # Step F
    print('layer{} total comparation output: {}, it should be: {}'.format(
            i+1, np.sum(compare_output), np.prod(compare_output.shape)))