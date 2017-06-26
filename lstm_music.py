from keras.models import Sequential
from keras.layers import Conv1D,MaxPooling1D,Dense,LSTM


def model_music(x, y, val_pct):
    """Construct the model to generate music using LSTM network"""
    # 数据相关参数
    (nsamples,input_length,output_dim) = x.shape
    n = int(nsamples*(1-val_pct))
    x_train = x[:n, :, :]
    y_train = y[:n]
    x_test = x[n:, :, :]
    y_test = y[n:]
    # 网络结构相关参数
    unit_num=128
    kernel_size=5
    pool_size=3
    lstm_size=128
    # 训练相关参数
    batch_size=64
    epochs=100

    model = Sequential()
    model.add(Conv1D(unit_num,kernel_size,activation='relu',input_shape=(input_length,output_dim)))
    model.add(MaxPooling1D(pool_size))
    model.add(LSTM(lstm_size))
    model.add(Dense(output_dim,activation='softmax'))

    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,verbose=2,validation_data=(x_test,y_test))
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('\nTest score:', score)
    print('Test accuracy:', acc)
    return model
