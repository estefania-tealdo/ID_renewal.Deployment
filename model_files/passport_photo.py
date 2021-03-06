import cv2
import os
import numpy as np
#from deeplab import *
from model_files.deeplab import *
from PIL import Image
import base64


#if __name__ == "__main__":
def passport_photo(img_bin):
    # parser = argparse.ArgumentParser(description="Let's do AI")
    # parser._action_groups.pop()
    # requiredNamed = parser.add_argument_group("required arguments")
    # requiredNamed.add_argument("-i", "--input-image-path", type=str, required=True)
    # requiredNamed.add_argument("-o", "--output-image-path", type=str, required=True)

    # optionalNamed = parser.add_argument_group("optional arguments")
    # optionalNamed.add_argument("--harr-weight-file", type=str, default="./haarcascade_frontalface_default.xml",
    #                            help="harr cascade weight file - https://github.com/opencv/opencv/tree/master/data"
    #                                 "/haarcascades")
    # optionalNamed.add_argument("--dim", type=int, default=600,
    #                            help="image dimension")
    # optionalNamed.add_argument("-s", "--semantic-segmentation-model", type=str, default="xception_coco_voctrainval",
    #                            help="model for segment human and background. Possible values:["
    #                                 "'mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', "
    #                                 "'xception_coco_voctrainaug', 'xception_coco_voctrainval'] ")
    # optionalNamed.add_argument("-m", "--model-dir", type=str, default="./Models")
    # args = parser.parse_args()

    # DEBUG = False
    #cascade classifier using harr weight
    

    #convert the b64 to jpg and save it
    with open("./model_files/in_pic.jpg", "wb") as fh:
        fh.write(base64.b64decode(img_bin))
    
    face_cascade = cv2.CascadeClassifier("./model_files/haarcascade_frontalface_default.xml")

    img = cv2.imread("./model_files/in_pic.jpg")
    (h, w) = img.shape[:2]
    if w >= h: #horizonal image
        height = 600
        ratio = height / h
        dim = (int(w * ratio), height)
    else: #vertical image
        width = 600
        ratio = width / w
        dim = (width, int(h * ratio))

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #find faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x_f, y_f, w_f, h_f) in [faces[0]]:
        img=img
        #if DEBUG:
            #cv2.rectangle(img, (x_f, y_f), (x_f + w_f, y_f + h_f), (255, 0, 0), 2)

    # crop to dimxdim and center user face in the image
    if w >= h:
        width_crop = (height - (w_f)) / 2
        if width_crop.is_integer():
            img = img[:, x_f - int(width_crop):x_f + w_f + int(width_crop)]
        else:
            img = img[:, x_f - int(np.floor(width_crop)):x_f + w_f + int(np.ceil(width_crop))]
    else:
        height_crop = (width - (h_f)) / 2
        if height_crop.is_integer():
            img = img[y_f - int(height_crop):y_f + h_f + int(height_crop), :]
        else:
            img = img[y_f - int(np.floor(height_crop)):y_f + h_f + int(np.ceil(height_crop)), :]

    #assert img.shape == (args.dim, args.dim, 3), "wrong shape:" + str(img.shape)

    #load pretrained segmentation model to get background
    # MODEL_NAME = args.semantic_segmentation_model
    # MODEL_DIR = args.model_dir
    # if not os.path.isdir(MODEL_DIR):
    #     os.mkdir(MODEL_DIR)
    # _TARBALL_NAME = MODEL_NAME + ".tar.gz"

    MODEL_NAME = 'xception_coco_voctrainval'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']
    MODEL_DIR = "./model_files/Models"
    """"_TARBALL_NAME = MODEL_NAME+'.tar.gz' """ 

    MODEL_NAME = 'xception_coco_voctrainval'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

    _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    _MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    }
    _TARBALL_NAME = 'deeplab_model.tar.gz'


    download_path = os.path.join(MODEL_DIR, _TARBALL_NAME)

    if not os.path.exists(download_path):
        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                               download_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(img)
    MODEL = DeepLabModel(download_path)
    resized_im, seg_map = MODEL.run(pil_im)

    #separate background and foreground of image.
    mask = Image.fromarray(create_pascal_label_colormap()[seg_map].astype(np.uint8))
    background = 255 * np.ones_like(mask).astype(np.uint8)
    foreground = np.array(resized_im.getdata()).reshape(resized_im.size[0], resized_im.size[1], 3).astype(float)
    background = background.astype(float)
    th, alpha = cv2.threshold(np.array(mask), 0, 255, cv2.THRESH_BINARY)
    alpha = cv2.GaussianBlur(alpha, (9, 9), 0)
    alpha = alpha.astype(float) / 255
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    outImage = cv2.add(foreground, background)

    #plt.imsave("./out_pic.jpg", outImage / 255)
    pil_im = Image.fromarray((outImage).astype(np.uint8))
    pil_im.save("./model_files/out_pic.jpg")

    #convert to b64 to send it thru a request
    with open("./model_files/out_pic.jpg", "rb") as img_file:
        out_b64=base64.b64encode(img_file.read())

    print(out_b64)
    return out_b64