from gaps_dataset import gaps
import os
import numpy as np
from PIL import Image

'''
You need to access below 
https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/german-asphalt-pavement-distress-dataset-gaps
to get a license.
Then, way how to utilize GAPs dataset is shared by official library, so please refer it.
'''

# download data
if os.path.exists('../downloaded_data/GAPs/v2/ZEB_50k_64'):
    print('GAPs data may already downloaded')
    pass
else:
    os.makedirs('../downloaded_data/GAPs', exist_ok=True)
    gaps.download(login='gapsro2s;i2A*7',
                output_dir='../downloaded_data/GAPs', 
                version=2,
                patchsize=224, 
                issue='ZEB_50k')


# prepare png data
if not os.path.exists('../downloaded_data/GAPs/v2/train/Potholes'):
    print('GAPs train_dataset may be ready')
    pass
else:
    cnt = [0, 0, 0, 0, 0, 0]
    classes = ['Intact_road', 'Applied_patches', 'Potholes', 'Inlaid_patches', 'Open_joints', 'Cracks']
    for i in classes:
        os.makedirs(os.path.join('../downloaded_data/GAPs/v2/train', i), exist_ok=True)


    x_train0, y_train0 = gaps.load_chunk(chunk_id=0, patchsize=224, issue='ZEB_50k', subset='train', datadir='../downloaded_data/GAPs')
    x_train1, y_train1 = gaps.load_chunk(chunk_id=1, patchsize=224, issue='ZEB_50k', subset='train', datadir='../downloaded_data/GAPs')
    x_train = np.concatenate([x_train0, x_train1])
    y_train = np.concatenate([y_train0, y_train1])
    import pdb
    pdb.set_trace()

    # dtype of original images is not uint8, for PIL library, change them to uint8 images
    for i in range(len(x_train)):
        nd_array = x_train[i][0]
        nd_array -= nd_array.min()
        scale = 255. / nd_array.max()
        img_array = np.uint8(np.round(nd_array*scale))
        img = Image.fromarray(img_array)

        cnt[int(y_train[i])] += 1
        img_path = '../downloaded_data/GAPs/v2/train/' + classes[int(y_train[i])] + '/' + str(cnt[int(y_train[i])])
        img.save(img_path+'.png')

if os.path.exists('../downloaded_data/GAPs/v2/test/Potholes'):
    print('GAPs test_dataset may be ready')
    pass
else:
    cnt = [0, 0, 0, 0, 0, 0]
    classes = ['Intact_road', 'Applied_patches', 'Potholes', 'Inlaid_patches', 'Open_joints', 'Cracks']
    for i in classes:
        os.makedirs(os.path.join('../downloaded_data/GAPs/v2/test', i), exist_ok=True)


    x_test, y_test = gaps.load_chunk(chunk_id=0, patchsize=224, issue='ZEB_50k', subset='test', datadir='../downloaded_data/GAPs')

    for i in range(len(x_test)):
        nd_array = x_test[i][0]
        nd_array -= nd_array.min()
        scale = 255. / nd_array.max()
        img_array = np.uint8(np.round(nd_array*scale))
        img = Image.fromarray(img_array)

        cnt[int(y_test[i])] += 1
        img_path = '../downloaded_data/GAPs/v2/test/' + classes[int(y_test[i])] + '/' + str(cnt[int(y_test[i])])
        img.save(img_path+'.png')
        print(i+1)