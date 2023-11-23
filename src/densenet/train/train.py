''' train howtord detection model with heyholoo dataset '''
import os
import argparse
import math
from datetime import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

def model_dense(input_shape):
    ''' In this function, the architecture of the model is specified '''
    model_d = DenseNet121(weights='imagenet',include_top=False, input_shape=input_shape)
    x = model_d.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(512,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    preds=Dense(3,activation='softmax')(x) #FC-layer

    model=Model(inputs=model_d.input,outputs=preds)
    return model


def prepare_data(train_csv, valid_csv):
    '''
    In this function, train and validation datasets are loaded from CSV files
    '''
    train_data=pd.read_pickle(train_csv)
    audio_paths = np.array(train_data['x'].tolist())
    classes = np.array(train_data['y'].tolist())

    audio_list = []
    for audio in audio_paths:
        audio_list.append(np.load(audio))
    audio_array =np.array(audio_list)

    labelencoder=LabelEncoder()
    classes=to_categorical(labelencoder.fit_transform(classes))

    x_train, y_train = audio_array, classes

    valid_data=pd.read_pickle(valid_csv)

    valid_audio_paths=np.array(valid_data['x'].tolist())
    valid_classes=np.array(valid_data['y'].tolist())

    valid_audio_list = []
    for v_audio in valid_audio_paths:
        valid_audio_list.append(np.load(v_audio))
    valid_audio_array =np.array(valid_audio_list)
    valid_classes=to_categorical(labelencoder.fit_transform(valid_classes))
    x_valid, y_valid = valid_audio_array, valid_classes

    return x_train, y_train, x_valid, y_valid


def main ():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir',type=str, required=True,help='path to save checkpoints.')
    parser.add_argument('--logdir',type=str, required=True,help='path to save logs')
    parser.add_argument('--train_csv', type=str, required=True, help='path to .csv file for train dataset')
    parser.add_argument('--valid_csv', type=str, required=True, help='path to .csv file for validation dataset')
    parser.add_argument('--name', type=str, required=False, default='model', help='save final model by this name')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    x_train, y_train, x_valid, y_valid = prepare_data(args.train_csv, args.valid_csv)
    n_batches = len(x_train)/16
    n_batches = math.ceil(n_batches)

    log_dir = os.path.join(args.logdir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    checkpoint_filepath = f'{args.checkpoint_dir}/ckpt'

    model_checkpoint_callback = ModelCheckpoint(checkpoint_filepath,
                                                save_weights_only=True,
                                                monitor='val_accuracy',
                                                mode='max',
                                                verbose=1,
                                                save_freq=50*n_batches)

    tensorboard_callback= TensorBoard(log_dir=log_dir)
    model = model_dense(input_shape = (100, 301, 3))

    model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=150, batch_size = 16, validation_data=(x_valid, y_valid),callbacks=[model_checkpoint_callback, tensorboard_callback])
    model.save(f'{args.checkpoint_dir}/{args.name}.h5')


if __name__=="__main__":
    main()
