import numpy as np
import h5py
from PIL import Image
import os

'''
Official BrainTumor dataset is released by MATLAB, so we need to
reconstruct to .png data.
'''

# If there is dataset already, skip this code
if os.path.exists('../downloaded_data/BrainTumor/1'):
    print('BurainTumor data may exist, please confirm')
    pass

else:
    # read mat file
    cross_validation_index = h5py.File('../downloaded_data/brain_mat/cvind.mat', 'r')
    cross_validation_index = np.array(cross_validation_index['cvind'])

    classes = ['meningioma', 'glioma', 'pituitary']
    num_each_class = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    for i in range(1, 6):
        for j in classes:
            os.makedirs(os.path.join('../downloaded_data/BrainTumor', str(i), j), exist_ok=True)

    for i in range(1, len(cross_validation_index[0])+1):
        f = h5py.File('../downloaded_data/brain_mat/'+str(i)+'.mat', 'r')
        label = f.get('cjdata/label')
        image_raw = f.get('cjdata/image')
        nd_image = np.array(image_raw)
        label = np.array(label)
        label = int(label[0][0]) - 1

        scale = 255. / (nd_image.max()-nd_image.min())
        nd_image = np.uint8(nd_image * scale)
        nd_image = (nd_image / (2**8-1))*255

        num_each_class[label][int(cross_validation_index[0][i-1])-1] += 1
        save_path = os.path.join('../downloaded_data/BrainTumor', str(int(cross_validation_index[0][i-1])), classes[label])
        pl_img = Image.fromarray(nd_image).convert('L')
        pl_img.save(os.path.join(save_path, str(num_each_class[label][int(cross_validation_index[0][i-1])-1])+'.png'))