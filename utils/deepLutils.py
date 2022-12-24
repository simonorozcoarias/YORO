import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K 


dicc_size={0:1000,1:3000,2:2000,3:2000,4:1000,5:1000,6:10000}
def loss_custom(y_true, y_pred):
    """
    loss_custom(y_true, y_pred)
    This function corresponds to the loss function that has 
    information about how bad is the prediction of the PRESENCE,
    POSITION AND LENGTH of a motiv in a DNA sequence. 
    This function contains some weights that give more relevance 
    to those predictions where there is indeed presence of the 
    domain. of the domain, since in this way the good learning of 
    the network is guaranteed.

    Parameters:
                y_true: tensor. Tensor that indicates presence, 
                position and length of motive in DNA secuences

                y_pred: tensor. Tensor with presence, position 
                and length predicition for motive in the DNA secuence

    Output: Loss function
    """
    focus = tf.gather(y_true,tf.constant([0]),axis=-1)
    w1=focus
    w2=(focus-1)*(-1)*(8/242)*3
    weights=tf.concat([w1+w2,focus,focus,focus, focus, focus, focus, focus, focus,focus,focus, focus, focus, focus, focus,focus,focus, focus, focus, focus, focus,focus,focus],axis=-1)
    salida = K.sum(K.pow((y_true-y_pred),2)*weights)
    return salida

def loss_precision_training(y_true, y_pred):
    presence_true = tf.gather(y_true,tf.constant([0]),axis=-1)
    presence_pred = tf.gather(y_pred,tf.constant([0]),axis=-1)
    salida = K.sum(presence_true*presence_pred)/K.sum(presence_pred)
    return salida

def loss_domains(y_true, y_pred):
    focus = tf.gather(y_true,tf.constant([0]),axis=-1)
    w1=focus
    w2=(focus-1)*(-1)*(8/(242))
    y_true = tf.gather(y_true,tf.constant([0,1,2,3,4,5,6,7,8,9]),axis=-1)
    y_pred = tf.gather(y_pred,tf.constant([0,1,2,3,4,5,6,7,8,9]),axis=-1)
    weights=tf.concat([w1+w2,focus,focus,focus, focus, focus, focus, focus, focus,focus],axis=-1)
    salida = K.sum(K.pow((y_true-y_pred),2)*weights)
    return salida

def loss_global(y_true, y_pred):
    """
    loss_global(y_true, y_pred)
    Esta función corresponde a la función de perdida que tiene información de 
    qué tan mal va la predicción de la PRESENCIA, POSICIÓN Y LONGITUD de un mo-
    tivo en una secuencia de ADN.

    Parameters:
                y_true: tensor. Tensor que indica la presencia, posición y lon-
                gitud del motivo en la sencuencia de ADN.

                y_pred: tensor. Tensor con la predicción de la presencia, 
                posición y longitud del motivo en la secuencia de ADN.

    Output: Función de pérdida
    """

    salida = K.sum(K.pow((y_true-y_pred),2))
    return salida

def YOLO_domain_JusAP(optimizador=Adam,lr=0.001,init_mode='glorot_normal',fun_act='linear',regularizer=l2,w_reg=0,lfun=loss_custom,n=32,ventana=2000):
    tf.keras.backend.clear_session()
    w = 200
    #n = 8
    inputs = tf.keras.Input(shape=(4,ventana, 1), name="input_1")
    L1 = tf.keras.layers.Conv2D(n, (4, 50), strides=(1,1),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(inputs)
    L1 = tf.keras.layers.ZeroPadding2D(padding=((0,0), (0,49)))(L1)
    L1 = tf.keras.layers.Conv2D(n, (1, 5), strides=(1,5),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L1)
    L1 = tf.keras.layers.BatchNormalization()(L1)
    L1 = tf.keras.layers.ReLU()(L1)
    L2 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L1)
    L2 = tf.keras.layers.BatchNormalization()(L2)
    L2 = tf.keras.layers.ReLU()(L2)
    L3 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L2)
    L3 = tf.keras.layers.BatchNormalization()(L3)
    L3 = tf.keras.layers.ReLU()(L3)
    L4 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L3)
    L4 = tf.keras.layers.BatchNormalization()(L4)
    L4 = Add()([L4,L1])
    L4 = tf.keras.layers.ReLU()(L4)

    L5 = tf.keras.layers.Conv2D(n, (1, 2), strides=(1,2),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L4)
    L5 = tf.keras.layers.BatchNormalization()(L5)
    L5 = tf.keras.layers.ReLU()(L5)
    L6 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L5)
    L6 = tf.keras.layers.BatchNormalization()(L6)
    L6 = tf.keras.layers.ReLU()(L6)
    L7 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L6)
    L7 = tf.keras.layers.BatchNormalization()(L7)
    L7 = tf.keras.layers.ReLU()(L7)
    L8 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L7)
    L8 = tf.keras.layers.BatchNormalization()(L8)
    L8 = Add()([L5,L8])
    L8 = tf.keras.layers.ReLU()(L8)

    L9 = tf.keras.layers.Conv2D(n, (1, 5), strides=(1,5),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L8)
    L9 = tf.keras.layers.BatchNormalization()(L9)
    L9 = tf.keras.layers.ReLU()(L9)
    L10 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L9)
    L10 = tf.keras.layers.BatchNormalization()(L10)
    L10 = tf.keras.layers.ReLU()(L10)
    L11 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L10)
    L11 = tf.keras.layers.BatchNormalization()(L11)
    L11 = tf.keras.layers.ReLU()(L11)
    L12 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L11)
    L12 = tf.keras.layers.BatchNormalization()(L12)
    L12 = Add()([L9,L12])
    L12 = tf.keras.layers.ReLU()(L12)

    L13 = tf.keras.layers.Conv2D(n, (1, 2), strides=(1,2),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L12)
    L13 = tf.keras.layers.BatchNormalization()(L13)
    L13 = tf.keras.layers.ReLU()(L13)
    L14 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L13)
    L14 = tf.keras.layers.BatchNormalization()(L14)
    L14 = tf.keras.layers.ReLU()(L14)
    L15 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L14)
    L15 = tf.keras.layers.BatchNormalization()(L15)
    L15 = tf.keras.layers.ReLU()(L15)
    L16 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L15)
    L16 = tf.keras.layers.BatchNormalization()(L16)
    L16 = Add()([L13,L16])
    L16 = tf.keras.layers.ReLU()(L16)
    
    layers = tf.keras.layers.Conv2D(3, (1, 10), strides=(1,1),padding='same',activation='sigmoid', use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L16)
    model = tf.keras.Model(inputs = inputs, outputs=layers)
    opt = optimizador(learning_rate=lr)
    loss_fn = lfun
    model.compile(loss=loss_fn, optimizer=opt, metrics=[loss_global])
    return model

def YOLO_domain_v14(optimizador=Adam,lr=0.001,init_mode='glorot_normal',fun_act='linear',regularizer=l2,w_reg=0,lfun=loss_custom,n=16,ventana=50000):
    tf.keras.backend.clear_session()
    momen = 0
    dp = 0.2
    w = 40
    #n = 16
    inputs = tf.keras.Input(shape=(4,ventana, 1), name="input_1")
    L1 = tf.keras.layers.Conv2D(n, (4, 50), strides=(1,1),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(inputs)
    L1 = tf.keras.layers.ZeroPadding2D(padding=((0,0), (0,49)))(L1)
    L1 = tf.keras.layers.Conv2D(n, (1, 5), strides=(1,5),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L1)
    L1 = tf.keras.layers.BatchNormalization()(L1)
    L1 = tf.keras.layers.ReLU()(L1)
    L2 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L1)
    L2 = tf.keras.layers.BatchNormalization()(L2)
    L2 = tf.keras.layers.ReLU()(L2)
    L3 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L2)
    L3 = tf.keras.layers.BatchNormalization()(L3)
    L3 = tf.keras.layers.ReLU()(L3)
    L4 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L3)
    L4 = tf.keras.layers.BatchNormalization()(L4)
    L4 = Add()([L4,L1])
    L4 = tf.keras.layers.ReLU()(L4)

    w = 20
    L5 = tf.keras.layers.Conv2D(n, (1, 2), strides=(1,2),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L4)
    L5 = tf.keras.layers.BatchNormalization()(L5)
    L5 = tf.keras.layers.ReLU()(L5)
    L6 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L5)
    L6 = tf.keras.layers.BatchNormalization()(L6)
    L6 = tf.keras.layers.ReLU()(L6)
    L7 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L6)
    L7 = tf.keras.layers.BatchNormalization()(L7)
    L7 = tf.keras.layers.ReLU()(L7)
    L8 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L7)
    L8 = tf.keras.layers.BatchNormalization()(L8)
    L8 = Add()([L5,L8])
    L8 = tf.keras.layers.ReLU()(L8)

    w = 4
    L9 = tf.keras.layers.Conv2D(n, (1, 5), strides=(1,5),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L8)
    L9 = tf.keras.layers.BatchNormalization()(L9)
    L9 = tf.keras.layers.ReLU()(L9)
    L10 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L9)
    L10 = tf.keras.layers.BatchNormalization()(L10)
    L10 = tf.keras.layers.ReLU()(L10)
    L11 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L10)
    L11 = tf.keras.layers.BatchNormalization()(L11)
    L11 = tf.keras.layers.ReLU()(L11)
    L12 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L11)
    L12 = tf.keras.layers.BatchNormalization()(L12)
    L12 = Add()([L9,L12])
    L12 = tf.keras.layers.ReLU()(L12)

    w = 2
    L13 = tf.keras.layers.Conv2D(n, (1, 2), strides=(1,2),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L12)
    L13 = tf.keras.layers.BatchNormalization()(L13)
    L13 = tf.keras.layers.ReLU()(L13)
    L14 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L13)
    L14 = tf.keras.layers.BatchNormalization()(L14)
    L14 = tf.keras.layers.ReLU()(L14)
    L15 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L14)
    L15 = tf.keras.layers.BatchNormalization()(L15)
    L15 = tf.keras.layers.ReLU()(L15)
    L16 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L15)
    L16 = tf.keras.layers.BatchNormalization()(L16)
    L16 = Add()([L13,L16])
    L16 = tf.keras.layers.ReLU()(L16)

    layers = tf.keras.layers.Conv2D(10, (1, 10), strides=(1,1),padding='same',activation='sigmoid', use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L16)
    model = tf.keras.Model(inputs = inputs, outputs=layers)
    opt = optimizador(learning_rate=lr)
    loss_fn = lfun
    model.compile(loss=loss_fn, optimizer=opt)#, metrics=[loss_domains])
    return model

def YOLO_domain_v15(optimizador=Adam,lr=0.001,init_mode='glorot_normal',fun_act='linear',regularizer=l2,w_reg=0,lfun=loss_domains,n=16,ventana=50000):
    tf.keras.backend.clear_session()
    momen = 0
    dp = 0.2
    w = 100
    #n=16
    inputs = tf.keras.Input(shape=(4,ventana, 1), name="input_1")
    L1 = tf.keras.layers.Conv2D(n, (4, 50), strides=(1,1),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(inputs)
    L1 = tf.keras.layers.ZeroPadding2D(padding=((0,0), (0,49)))(L1)
    L1 = tf.keras.layers.Conv2D(n, (1, 5), strides=(1,5),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L1)
    L1 = tf.keras.layers.BatchNormalization()(L1)
    L1 = tf.keras.layers.ReLU()(L1)
    L2 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L1)
    L2 = tf.keras.layers.BatchNormalization()(L2)
    L2 = tf.keras.layers.ReLU()(L2)
    L3 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L2)
    L3 = tf.keras.layers.BatchNormalization()(L3)
    L3 = tf.keras.layers.ReLU()(L3)
    L4 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L3)
    L4 = tf.keras.layers.BatchNormalization()(L4)
    L4 = Add()([L4,L1])
    L4 = tf.keras.layers.ReLU()(L4)

    w = 50
    L5 = tf.keras.layers.Conv2D(n, (1, 2), strides=(1,2),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L4)
    L5 = tf.keras.layers.BatchNormalization()(L5)
    L5 = tf.keras.layers.ReLU()(L5)
    L6 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L5)
    L6 = tf.keras.layers.BatchNormalization()(L6)
    L6 = tf.keras.layers.ReLU()(L6)
    L7 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L6)
    L7 = tf.keras.layers.BatchNormalization()(L7)
    L7 = tf.keras.layers.ReLU()(L7)
    L8 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L7)
    L8 = tf.keras.layers.BatchNormalization()(L8)
    L8 = Add()([L5,L8])
    L8 = tf.keras.layers.ReLU()(L8)

    w = 10
    L9 = tf.keras.layers.Conv2D(n, (1, 5), strides=(1,5),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L8)
    L9 = tf.keras.layers.BatchNormalization()(L9)
    L9 = tf.keras.layers.ReLU()(L9)
    L10 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L9)
    L10 = tf.keras.layers.BatchNormalization()(L10)
    L10 = tf.keras.layers.ReLU()(L10)
    L11 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L10)
    L11 = tf.keras.layers.BatchNormalization()(L11)
    L11 = tf.keras.layers.ReLU()(L11)
    L12 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L11)
    L12 = tf.keras.layers.BatchNormalization()(L12)
    L12 = Add()([L9,L12])
    L12 = tf.keras.layers.ReLU()(L12)

    w = 5
    L13 = tf.keras.layers.Conv2D(n, (1, 2), strides=(1,2),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L12)
    L13 = tf.keras.layers.BatchNormalization()(L13)
    L13 = tf.keras.layers.ReLU()(L13)
    L14 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L13)
    L14 = tf.keras.layers.BatchNormalization()(L14)
    L14 = tf.keras.layers.ReLU()(L14)
    L15 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L14)
    L15 = tf.keras.layers.BatchNormalization()(L15)
    L15 = tf.keras.layers.ReLU()(L15)
    L16 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L15)
    L16 = tf.keras.layers.BatchNormalization()(L16)
    L16 = Add()([L13,L16])
    L16 = tf.keras.layers.ReLU()(L16)

    layers = tf.keras.layers.Conv2D(10, (1, 10), strides=(1,1),padding='same',activation='sigmoid', use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L16)
    model = tf.keras.Model(inputs = inputs, outputs=layers)
    opt = optimizador(learning_rate=lr)
    loss_fn = lfun
    model.compile(loss=loss_fn, optimizer=opt)#, metrics=[loss_domains])
    return model

def YOLO_domain_v17(optimizador=Adam,lr=0.001,momen=0,init_mode='glorot_normal',fun_act='linear',dp=0.2,regularizer=l2,w_reg=0,ventana=50000):
    tf.keras.backend.clear_session()
    w = 100
    n = 16
    inputs = tf.keras.Input(shape=(4,ventana, 1), name="input_1")
    L1 = tf.keras.layers.Conv2D(n, (4, 50), strides=(1,1),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(inputs)
    L1 = tf.keras.layers.ZeroPadding2D(padding=((0,0), (0,49)))(L1)
    L1 = tf.keras.layers.Conv2D(n, (1, 5), strides=(1,5),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L1)
    L1 = tf.keras.layers.BatchNormalization()(L1)
    L1 = tf.keras.layers.ReLU()(L1)
    L2 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L1)
    L2 = tf.keras.layers.BatchNormalization()(L2)
    L2 = tf.keras.layers.ReLU()(L2)
    L3 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L2)
    L3 = tf.keras.layers.BatchNormalization()(L3)
    L3 = tf.keras.layers.ReLU()(L3)
    L4 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L3)
    L4 = tf.keras.layers.BatchNormalization()(L4)
    L4 = Add()([L4,L1])
    L4 = tf.keras.layers.ReLU()(L4)

    w = 50
    L5 = tf.keras.layers.Conv2D(n, (1, 2), strides=(1,2),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L4)
    L5 = tf.keras.layers.BatchNormalization()(L5)
    L5 = tf.keras.layers.ReLU()(L5)
    L6 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L5)
    L6 = tf.keras.layers.BatchNormalization()(L6)
    L6 = tf.keras.layers.ReLU()(L6)
    L7 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L6)
    L7 = tf.keras.layers.BatchNormalization()(L7)
    L7 = tf.keras.layers.ReLU()(L7)
    L8 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L7)
    L8 = tf.keras.layers.BatchNormalization()(L8)
    L8 = Add()([L5,L8])
    L8 = tf.keras.layers.ReLU()(L8)

    w = 10
    L9 = tf.keras.layers.Conv2D(n, (1, 5), strides=(1,5),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L8)
    L9 = tf.keras.layers.BatchNormalization()(L9)
    L9 = tf.keras.layers.ReLU()(L9)
    L10 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L9)
    L10 = tf.keras.layers.BatchNormalization()(L10)
    L10 = tf.keras.layers.ReLU()(L10)
    L11 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L10)
    L11 = tf.keras.layers.BatchNormalization()(L11)
    L11 = tf.keras.layers.ReLU()(L11)
    L12 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L11)
    L12 = tf.keras.layers.BatchNormalization()(L12)
    L12 = Add()([L9,L12])
    L12 = tf.keras.layers.ReLU()(L12)

    w = 5
    L13 = tf.keras.layers.Conv2D(n, (1, 2), strides=(1,2),activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L12)
    L13 = tf.keras.layers.BatchNormalization()(L13)
    L13 = tf.keras.layers.ReLU()(L13)
    L14 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L13)
    L14 = tf.keras.layers.BatchNormalization()(L14)
    L14 = tf.keras.layers.ReLU()(L14)
    L15 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L14)
    L15 = tf.keras.layers.BatchNormalization()(L15)
    L15 = tf.keras.layers.ReLU()(L15)
    L16 = tf.keras.layers.Conv2D(n, (1, w), strides=(1,1),padding='same',activation=fun_act, use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L15)
    L16 = tf.keras.layers.BatchNormalization()(L16)
    L16 = Add()([L13,L16])
    L16 = tf.keras.layers.ReLU()(L16)

    layers = tf.keras.layers.Conv2D(9, (1, 10), strides=(1,1),padding='same',activation='sigmoid', use_bias=True, kernel_initializer=init_mode, bias_initializer='zeros', kernel_regularizer=regularizer(w_reg), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(L16)
    model = tf.keras.Model(inputs = inputs, outputs=layers)
    opt = optimizador(learning_rate=lr)
    model.compile(loss=loss_domains, optimizer=opt, metrics=[loss_precision_training])
    return model

def loadNNArchitecture(ventana,weigths_file):
    model = YOLO_domain_v17(ventana=ventana)
    model.load_weights(weigths_file)
    print("Model loaded succesfully!!")
    return model

def IOU(box1,box2,size1,size2):
    """
    IOU(box1,box2)
    Esta función calcula el valor de Intersection Over Union para dos motivos
    predichos sobre la misma secuencia de ADN.

    Parameters:
                box1: list. Lista con tres valores: (1) posición relativa del 
                motivodentro de la cuadricula de la grilla, (2) longitud norma-
                lizada del motivo, (3) número de la cuadricula.

                box2: list. Lista con tres valores: (1) posición relativa del 
                motivodentro de la cuadricula de la grilla, (2) longitud norma-
                lizada del motivo, (3) número de la cuadricula.

    Output: Intersection over union
    """

    pi1,len1,n1 = box1
    pi2,len2,n2 = box2

    pi1=(pi1+n1)*100
    pf1=pi1+len1*size1
    pi2=(pi2+n2)*100
    pf2=pi2+len2*size2
    xi1 = max([pi1,pi2])
    xi2 = min([pf1,pf2])
    inter_width = xi2-xi1
    inter_area = max([inter_width,0])
    box1_area = len1*size1
    box2_area = len2*size2
    union_area = box1_area+box2_area-inter_area
    iou = inter_area/(union_area + K.epsilon())
    return iou

def NMS(Yhat, threshold_presence, threshold_NMS):
  """
  NMS(Yhat, threshold_presence, threshold_NMS)
  Esta función realiza la operación del non-Max Suppression para las prediccio-
  nes que arroja el modelo YOLO.

  Parameters:
              Yhat: array. Contiene las predicciones hechas por el modelo YOLO.

              threshold_presence: float. Umbral por encima del cual se considera 
              que una probabilidad apunta a que efectivamente haya presencia del 
              motivo a identificar.

              threshold_NMS: float. Umbral por encima del cual se considera que
              dos motivos contiguos predichos hacen referencia al mismo motivo.

  Output:   Array con las predicciones del modelo YOLO una vez que se han 
  eliminado las predicciones irrelevantes.
  """

  Yhat_new = np.copy(Yhat)
  for index in range(Yhat.shape[0]):
    mascara = (Yhat[index,:,:,0:1]>=threshold_presence)*1
    data_pred = mascara*Yhat[index,:,:,:]
    data_mod = np.copy(data_pred[0,:,0])
    cont=1
    while cont>0:
      try:
        ind_first = np.nonzero(data_mod)[0][0]
      except:
        break
      ind_nonzero = np.nonzero(data_mod)[0][1:]
      for i in ind_nonzero:
          box1=[data_pred[0,ind_first,1],data_pred[0,ind_first,2],ind_first]
          box2=[data_pred[0,i,1],data_pred[0,i,2],i]
          size1=dicc_size[np.argmax(data_pred[0,ind_first,3:])]
          size2=dicc_size[np.argmax(data_pred[0,i,3:])]
          iou = IOU(box1,box2,size1,size2)
          if iou>=threshold_NMS:
            if data_mod[i]>data_mod[ind_first]:
              data_pred[0,ind_first,:]=0
              data_mod[ind_first]=0
              break
            else:
              data_pred[0,i,:]=0
              data_mod[i]=0
          else:
            data_mod[ind_first]=0
            break
      cont=np.sum(ind_nonzero)
    Yhat_new[index,:,:,:]=data_pred
  return Yhat_new

def BEE(weights,window,n,x):
    inputs = tf.constant(x,dtype=tf.float32)
    W=tf.constant(weights,dtype=tf.float32)
    b=tf.constant(-39*np.ones((1,n)).reshape(n,),dtype=tf.float32)

    layers = tf.nn.conv2d(inputs, W, strides=[1,1],padding='VALID')
    layers =tf.nn.bias_add(layers,b)
    layers_full = tf.nn.relu(layers)

    layers=tf.math.reduce_sum(layers_full,axis=-1,keepdims=True)
    layers = tf.pad(layers,tf.constant([[0,0],[0,0], [0,49],[0,0]]))
    layers = tf.nn.max_pool2d(layers,ksize=[1, 50],strides=[1,50],padding='VALID')
    
    layers = tf.nn.max_pool2d(layers,ksize=(1, 5),strides=[1,5],padding='VALID')
    model = [layers.numpy(),layers_full.numpy()]
    return model

def position_LTR(tensor):
    vector = np.zeros((tensor.shape[0]+2,))
    vector[1:-1] = tensor
    vector = (vector[1:]>0)*1-(vector[0:-1]>0)*1 
    indices = np.nonzero(vector)[0]
    indices_LTR=[(indices[-2])*250,(indices[-1])*250]

    return indices_LTR

def position_LTR_right(tensor_full,inicio):
    indice_inicio = np.nonzero(tensor_full[0,0,inicio-10:inicio+300,:])
    return indice_inicio[1][0]*50

def index_pos(y):
  indices_start=[]
  indices_end=[]
  longitudes=[]
  posiciones = np.absolute(y[1:]-y[0:-1])
  vector = np.nonzero(posiciones)[0]
  
  if len(vector)%2!=0:
    vector=np.append(vector,np.array([49999]))
  for i in range(len(vector)):
    if i%2==0:
      indices_start.append(vector[i])
    else:
      indices_end.append(vector[i])
      longitudes.append(vector[i]-vector[i-1])
  return indices_start,indices_end,longitudes

def nt_TE(y,ventana,sample,threshold_presence,distancia=30):
  nucleotidos = np.zeros((sample,1,ventana))
  mask = (y[:,:,:,0:1]>=threshold_presence)*1
  valores =[]
  for i in range(y.shape[0]):
    indices = np.nonzero(mask[i,0,:,0])[0]
    for h in range(len(indices)):
      size = dicc_size[np.nonzero(y[i,0,indices[h],3:]==np.amax(y[i,0,indices[h],3:]))[0][0]]
      if h!=0 and (indices[h]-indices[h-1])<distancia:
        j1=indices[h-1]
        j2=indices[h]
        inicio = int(j1*100+y[i,0,j1,1]*100)
        inicio2=int(j2*100+y[i,0,j2,1]*100)
        fin = int(inicio2+y[i,0,j2,2]*size)
        nucleotidos[i,0,inicio:fin]=1
  return nucleotidos

def label_LTR(X_test,y,threshold_presence):
    ventana = int(y.shape[2]*100)
    label = np.zeros((y.shape[0],y.shape[1],y.shape[2],3))
    y_nt = nt_TE(y,ventana,y.shape[0],threshold_presence)
    cont_left=0
    cont_right=0
    for i in range(y.shape[0]):
      indices_start,indices_end,_ = index_pos(y_nt[i,0,:])
      for k in range(len(indices_start)):
        if k==0:
          valor_comienzo_seq = 10000
        else:
          valor_comienzo_seq = min([10000,int((indices_start[k]-indices_end[k-1])/(50*3/2))*50])
        comienzo =indices_start[k]-valor_comienzo_seq
        if comienzo<0:
          comienzo=0
        if k+1==len(indices_start):
          valor_inter_seq = 200
        else:
          valor_inter_seq = int((indices_start[k+1]-indices_end[k])/(50*3/2))
        n=min([200,valor_inter_seq,int((50000-indices_end[k])/50)])
        if n==0:
          continue
        ind = indices_end[k]
        weights = X_test[i,:,ind:ind+(n)*50].reshape((4,n,1,50)).transpose((0,3,2,1))
        window = indices_start[k]-comienzo
        if window < 50:
          continue
        tensor, tensor_full = BEE(weights,window,n,X_test[i:i+1,:,comienzo:indices_start[k]].reshape((1,4,window,1)))
        tensor = tensor[0,0,:,0]
        try:
          inicio,fin = position_LTR(tensor)
        except:
          cont_left+=1
          continue
        try:
          inicio_right= position_LTR_right(tensor_full,inicio)
        except:
          cont_right+=1
          continue
          
        inicio=inicio+comienzo
        fin=fin+comienzo
        inicio_right = inicio_right+indices_end[k]
        longitud = fin-inicio
        label[i,0,int(inicio/100),0]=1
        label[i,0,int(inicio/100),1] = (inicio-int((inicio)/100)*100)/100
        label[i,0,int(inicio/100),2] = longitud/10000

        label[i,0,int(inicio_right/100),0]=1
        label[i,0,int(inicio_right/100),1] = (inicio_right-int((inicio_right)/100)*100)/100
        label[i,0,int(inicio_right/100),2] = longitud/10000
        if np.sum(label[i,0,:,0])%2!=0:
          label[i,0,:,0]=0
    #print(cont_left)
    #print(cont_right)
    return label
  
