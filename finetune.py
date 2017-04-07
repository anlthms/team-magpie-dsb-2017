import pandas as pd
import numpy as np
import sys
import os
import dicom
from sklearn.ensemble import GradientBoostingClassifier


nfeats = 1

def log_loss(t, y):
    assert t.shape == y.shape
    eps = 1e-15
    y = np.clip(y, eps, 1-eps)
    return -np.mean(t*np.log(y) + (1 - t)*np.log(1 - y))

def get_features(df, set_name):
    cache_name = set_name + '-features.npy'
    if os.path.exists(cache_name):
        features = np.load(cache_name)
        return features[:, :nfeats]

    features = np.zeros((df.shape[0], nfeats), dtype=np.float32)
    for idx, row in df.iterrows():
        uid = row['id']
        path = os.path.join(data_dir, 'stage1', uid)
        files = os.listdir(path)
        slices = [dicom.read_file(path + '/' + f) for f in files]
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        for s in slices:
            s.SliceThickness = slice_thickness
        features[idx, 0] = slice_thickness
        features[idx, 1] = s.RescaleIntercept
        features[idx, 2] = len(files)

        print uid, np.int32(np.round(features[idx]))
    np.save(cache_name, features)
    return features[:, :nfeats]


data_dir = sys.argv[1]
LABELS_FILE =  os.path.join(data_dir, 'stage1_labels.csv')

all_labels = pd.read_csv(LABELS_FILE)
val_size = all_labels.shape[0] // 10

train_labels = np.zeros((all_labels.shape[0] - val_size, 1), dtype=np.float32)
val_labels = np.zeros((val_size, 1), dtype=np.float32)

train_data = np.zeros((train_labels.shape[0], 1 + nfeats), dtype=np.float32)
val_data = np.zeros((val_labels.shape[0], 1 + nfeats), dtype=np.float32)

all_labels = all_labels.sample(frac=1, random_state=0).reset_index(drop=True)
train_rows = all_labels.iloc[0:train_labels.shape[0]].reset_index(drop=True)
val_rows = all_labels.iloc[train_labels.shape[0]:].reset_index(drop=True)

assert val_rows.shape[0] == val_size

start = 0
for idx in range(1, 10):
    preds = pd.read_csv('val-predictions-' + str(idx) + '.csv')
    labels = pd.read_csv('val-labels-' + str(idx) + '.csv')
    print('idx %d loss %.4f' % (idx, log_loss(labels, preds)))
    end = start + labels.shape[0]
    train_data[start:end, 0] = preds.values.ravel()
    train_labels[start:end] = labels.values
    start = end

assert end == train_labels.shape[0]
preds = pd.read_csv('val-predictions-' + str(0) + '.csv')
labels = pd.read_csv('val-labels-' + str(0) + '.csv')
val_data[:, 0] = preds.values.ravel()
val_labels[:] = labels.values

print('idx %d loss %.4f' % (0, log_loss(labels.values.ravel(), preds.values.ravel())))

train_data[:, 1:] = get_features(train_rows, 'train')
val_data[:, 1:] = get_features(val_rows, 'val')

rng = np.random.RandomState(0)
model = GradientBoostingClassifier(n_estimators=500, max_depth=1,
                                   random_state=rng, verbose=1)
model.fit(train_data, train_labels.ravel())
val_preds = model.predict_proba(val_data)[:, 1]

print('log loss before %.4f' % log_loss(val_labels.ravel(), val_data[:, 0]))
print('log loss after %.4f' % log_loss(val_labels.ravel(), val_preds))

test_preds = pd.read_csv('predictions.csv')
test_data = np.zeros((test_preds.shape[0], 1 + nfeats), dtype=np.float32)

test_data[:, 0] = test_preds['cancer'].values.ravel()
test_data[:, 1:] = get_features(test_preds, 'test')
new_test_preds = model.predict_proba(test_data)[:, 1]
ids = test_preds['id'].values.ravel()
df = pd.DataFrame({'id': pd.Series(ids), 'cancer': pd.Series(np.squeeze(new_test_preds))})
df.to_csv('tuned_predictions.csv', header=True, columns=['id','cancer'], index=False)
