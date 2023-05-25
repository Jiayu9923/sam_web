import onnxruntime

ort_session = onnxruntime.InferenceSession('sam_onnx.onnx')

# Get input names
input_names = [input.name for input in ort_session.get_inputs()]
print('Input names: ', input_names)

# Get input shapes
input_shapes = [input.shape for input in ort_session.get_inputs()]
print('Input shapes: ', input_shapes)