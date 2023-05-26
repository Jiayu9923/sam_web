from flask import Flask, request, jsonify
from PIL import Image
import onnxruntime as rt
import numpy as np
import io
import base64
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
from io import BytesIO
from flask_cors import CORS




app = Flask(__name__)
CORS(app)

sess = rt.InferenceSession("model/sam_onnx.onnx")

@app.route('/segment', methods=['POST'])
def segment():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']\

    # Convert FileStorage to PIL Image
    pil_image = Image.open(file.stream).convert('RGB')

    # Convert PIL Image to OpenCV array (BGR)
    image = np.array(pil_image)
    # Convert RGB to BGR
    image = image[:, :, ::-1].copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # display image
    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # plt.axis('off')
    # plt.savefig('test.png')


    def show_anns(anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)
        # return img

    checkpoint = "model/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    print("Loading 1")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)

    print("Loading 2")
    mask_generator = SamAutomaticMaskGenerator(sam)

    print("Loading 3")
    masks = mask_generator.generate(image)

    print("Loading 4")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig('img/res.png')
    # plt.show()

    print("Loading 5")
    res = cv2.imread('img/res.png')
    is_success, buffer = cv2.imencode(".png", res)
    if not is_success:
        return jsonify({'error': 'Error during image conversion'}), 500

    print("Loading 6")
    # Convert buffer to byte stream
    byte_stream = BytesIO(buffer)

    print("Loading 7")
    # Convert byte stream to Base64
    base64_encoded_image = base64.b64encode(byte_stream.getvalue()).decode('utf-8')

    # Return as JSON
    return jsonify({'image': base64_encoded_image})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8181)
