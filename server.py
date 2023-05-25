from flask import Flask, request, jsonify
from PIL import Image
import onnxruntime as rt
import numpy as np
import io
import base64



app = Flask(__name__)

sess = rt.InferenceSession("sam_onnx.onnx")

@app.route('/segment', methods=['POST'])
def segment():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    orig_size = image.size
    image = np.array(image).astype('float32') / 255
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    result = sess.run([output_name], {input_name: image})[0]

    result = result[0] # remove batch dimension
    result = np.transpose(result, (1, 2, 0)) # move channel dimension to the end
    result = (result * 255).astype('uint8') # scale values back to [0, 255]

    output_image = Image.fromarray(result).resize(orig_size) # resize to original size
    output_image_byte_arr = io.BytesIO()
    output_image.save(output_image_byte_arr, format='PNG') # convert PIL Image to byte array
    encoded_image = base64.b64encode(output_image_byte_arr.getvalue()).decode('ascii') # encode as base64

    return jsonify({'image': encoded_image})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8181)
