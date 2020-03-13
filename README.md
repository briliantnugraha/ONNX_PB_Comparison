# ONNX_PB_Comparison
 This script contains Ways to add intermediate layer in ONNX, Get intermediate layer in Tensorflow frozen graph, and compare both intermediate layer outputs

***There are three parts:***

## A. Get intermediate layer in ONNX, Please check the following instruction inside of the script
```
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
```

## B. Get intermediate layer of Frozen Inference Graph of Tensorflow

```
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
```

## C. Compare the intermediate layer results of ONNX and PB layers

```
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
```

## Required libraries:
```
onnxruntime
numpy
cv2
os
onnx
tensorflow
```

Task:
- [x] Get intermediate layer in ONNX
- [x] Get intermediate layer of Frozen Inference Graph of Tensorflow
- [x] Compare the intermediate layer results of ONNX and PB layers
