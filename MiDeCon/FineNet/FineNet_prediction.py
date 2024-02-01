from __future__ import absolute_import
from __future__ import division

import sys, os
sys.path.append(os.path.realpath('../CoarseNet'))
sys.path.append(os.path.abspath('../'))
print('Path variables:', sys.path)
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras import backend as K
import tensorflow as tf
# Suppress tensorflow warnings for now
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import argparse
import numpy as np
import json
import traceback
import imageio.v2 as imageio

from CoarseNet.MinutiaeNet_utils import *
from CoarseNet.CoarseNet_utils import *
from CoarseNet.CoarseNet_model import *


config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)
K.set_session(sess)


# Prepare dataset for testing.
inference_set =['../../Data/testsamples/sample1/']

 
#image file extension 
extens = '.bmp'
#path to minutiae files
minupath = "../../Data/testsamples"
#output director
db_name = inference_set[0].split('/')[3]
output_dir = '../output_FineNet/predictions_on_%s' % (db_name)

if not os.path.exists(output_dir + '/'):
    mkdir(output_dir + '/')

#path of trained finenet
FineNet_path = '../output_FineNet/FineNet_dropout/FineNet__dropout__model.h5'

logging = init_log(output_dir)

# If use FineNet to refine, set into True
isHavingFineNet = True

minus_arr = []
for nn in range(100):
    minus_arr.append(str(-1))

# ====== Load FineNet to verify
if isHavingFineNet == True:
    model = FineNetmodel(num_classes=2,
                         pretrained_path=None,
                         input_shape=(224,224,3))

    dense = model.layers[-1]
    model_out = Model(model.input, model.layers[-2].output)
    model_out.summary()
    x = model_out.output
    dropout = Dropout(rate=0.3)(x, training=True)
    prediction = dense(dropout)
    model_FineNet = Model(inputs=model.input, outputs=prediction) 
    
    model_FineNet.summary()
    if FineNet_path != None:
        model_FineNet.load_weights(FineNet_path)
        print('Pretrained FineNet loaded.')

    model_FineNet.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0),
                  metrics=['accuracy'])
 
#returns minutia patch 
def getpatch(x, y, patch_minu_radio):
    try:
        x_patch_minu_radio = patch_minu_radio
        y_patch_minu_radio = patch_minu_radio
        # Extract patch from image
        x_begin = x - patch_minu_radio
        y_begin = y - patch_minu_radio
        
        #check if begin out of image
        if x_begin < 0:
            x_patch_minu_radio += ((-1)*x_begin)
            x_begin = 0
        if y_begin < 0:
            y_patch_minu_radio += ((-1)*y_begin)
            y_begin = 0
        #check if end out of image
        x_end = x_begin + 2 * x_patch_minu_radio
        y_end = y_begin + 2 * y_patch_minu_radio
        if x_end > img_size[0]:
            offset = x_end - img_size[0]
            x_begin -= offset
            x_end = img_size[0]
        if y_end > img_size[1]:
            offset = y_end - img_size[1]
            y_begin -= offset
            y_end = img_size[1]
        
        #create patch
        patch_minu = original_image[x_begin:x_end, y_begin:y_end]
        return patch_minu
    except: return np.array(None)



for i, deploy_set in enumerate(inference_set):
    print("Set", deploy_set)
    set_name = deploy_set.split('/')[-2]

    mkdir(output_dir + '/'+ set_name + '/')
    
    # Read image and GT
    img_name, folder_name, img_size = get_maximum_img_size_and_names(deploy_set)

    logging.info("Predicting \"%s\":" % (set_name))


    for i in range(0, len(img_name)):
        logging.info("\"%s\" %d / %d: %s" % (set_name, i + 1, len(img_name), img_name[i]))

        image = imageio.imread(deploy_set + img_name[i] + extens, pilmode='L')# / 255.0
        
        img_size = image.shape
        img_size = np.array(img_size, dtype=np.int32) // 8 * 8
        image = image[:img_size[0], :img_size[1]]

        original_image = image.copy()
        
        ########read minutiae from file########
        minufile = open("%s/%s/%s/%s.mnt" % (minupath, set_name, "mnt_results", img_name[i]), 'r')
        minu_list = []
        for line, content in enumerate(minufile):
            if line > 1:
                x, y, _, _ = [float(x) for x in content.split()]
                minu_list.append([int(x), int(y)])
   
        preds = {}
        if isHavingFineNet == True:
            # ======= Verify using FineNet ============
            patch_minu_radio = 28
            if FineNet_path != None:
                for idx, minu in enumerate(minu_list):
                    print((minu[0], minu[1]))
                    minu_prediction = []
                    try:
                        # Extract patch from image
                        patch_minu = getpatch(int(minu[1]), int(minu[0]), patch_minu_radio)           
                        patch_minu = cv2.resize(patch_minu, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)

                        ret = np.empty((patch_minu.shape[0], patch_minu.shape[1], 3), dtype=np.uint8)
                        ret[:, :, 0] = patch_minu
                        ret[:, :, 1] = patch_minu
                        ret[:, :, 2] = patch_minu
                        patch_minu = ret
                        patch_minu = np.expand_dims(patch_minu, axis=0)
                        
                        #predict 100 times on each minutia
                        for n in range(100):
                                [isMinutiaeProb] = model_FineNet.predict(patch_minu)
                                isMinutiaeProb = isMinutiaeProb[0]

                                minu_prediction.append(str(isMinutiaeProb))
                                
                    except KeyboardInterrupt:
                        raise
                    except:
                        minu_prediction = minus_arr
                            
                    preds[str(idx)] = minu_prediction


        with open("%s/%s/%s.json" % (output_dir, set_name, img_name[i]), 'w') as file:
            json.dump(preds, file)
