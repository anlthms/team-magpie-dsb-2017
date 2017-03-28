from keras import backend as K
def dice_coef(y_true, y_pred):
    SMOOTH = 1.
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f,axis=-1)
    union = K.sum(y_true_f,axis=-1) + K.sum(y_pred_f,axis=-1)
    return K.mean((2. * intersection + SMOOTH) / (union + SMOOTH))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



