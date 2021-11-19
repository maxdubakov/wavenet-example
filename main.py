from datetime import timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate
from tensorflow.keras.optimizers import Adam


aapl_raw = pd.read_csv('./data/AAPL.csv')
msft_raw = pd.read_csv('./data/MSFT.csv')
sp500_raw = pd.read_csv('./data/^GSPC.csv')
nsdq_raw = pd.read_csv('./data/^IXIC.csv')

print(aapl_raw.info())

aapl_raw.index = pd.to_datetime(aapl_raw.Date).dt.date
msft_raw.index = pd.to_datetime(msft_raw.Date).dt.date
sp500_raw.index = pd.to_datetime(sp500_raw.Date).dt.date
nsdq_raw.index = pd.to_datetime(nsdq_raw.Date).dt.date

print(aapl_raw.head())

aapl_raw['aAC'] = aapl_raw['Adj Close']
msft_raw['mAC'] = msft_raw['Adj Close']
sp500_raw['sAC'] = sp500_raw['Adj Close']
nsdq_raw['nAC'] = nsdq_raw['Adj Close']

aapl_raw['aV'] = aapl_raw['Volume']
msft_raw['mV'] = msft_raw['Volume']
sp500_raw['sV'] = sp500_raw['Volume']
nsdq_raw['nV'] = nsdq_raw['Volume']

aapl = aapl_raw.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
msft = msft_raw.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
sp500 = sp500_raw.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
nsdq = nsdq_raw.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

df = pd.concat([sp500, nsdq, msft, aapl], axis=1)
print(df.head())

### MODEL
df = df.T

data_start_date = df.columns[1]
data_end_date = df.columns[-1]

pred_steps = 100
pred_length = timedelta(pred_steps)

first_day = pd.to_datetime(data_start_date)
last_day = pd.to_datetime(data_end_date)

val_pred_start = last_day - pred_length + timedelta(1)
val_pred_end = last_day

train_pred_start = val_pred_start - pred_length
train_pred_end = val_pred_start - timedelta(days=1)

enc_length = train_pred_start - first_day

train_enc_start = first_day
train_enc_end = train_enc_start + enc_length - timedelta(1)

val_enc_start = train_enc_start + pred_length
val_enc_end = val_enc_start + enc_length - timedelta(1)

print('Train encoding:', train_enc_start, '-', train_enc_end)
print('Train prediction:', train_pred_start, '-', train_pred_end, '\n')
print('Val encoding:', val_enc_start, '-', val_enc_end)
print('Val prediction:', val_pred_start, '-', val_pred_end)

print('\nEncoding interval:', enc_length.days)
print('Prediction interval:', pred_length.days)

date_to_index = pd.Series(index=pd.Index([pd.to_datetime(c) for c in df.columns[1:]]),
                          data=[i for i in range(len(df.columns[1:]))])

series_array = df[df.columns[1:]].values


def get_time_block_series(series_array, date_to_index, start_date, end_date):
    inds = date_to_index[start_date:end_date]
    return series_array[:, inds]


def transform_series_encode(series_array):
    series_array = np.log(series_array)
    series_mean = series_array.mean(axis=1).reshape(-1, 1)
    series_array = series_array - series_mean
    series_array = series_array.reshape((series_array.shape[0], series_array.shape[1], 1))

    return series_array, series_mean


def transform_series_decode(series_array, encode_series_mean):
    series_array = np.log(series_array)
    series_array = series_array - encode_series_mean
    series_array = series_array.reshape((series_array.shape[0], series_array.shape[1], 1))

    return series_array


n_filters = 32
filter_width = 2
dilation_rates = [2 ** i for i in range(7)] * 2

# define an input history series and pass it through a stack of dilated causal convolution blocks
history_seq = Input(shape=(None, 1))
x = history_seq

skips = []
for dilation_rate in dilation_rates:
    # preprocessing - equivalent to time-distributed dense
    x = Conv1D(16, 1, padding='same', activation='relu')(x)

    # filter
    x_f = Conv1D(filters=n_filters,
                 kernel_size=filter_width,
                 padding='causal',
                 dilation_rate=dilation_rate)(x)

    # gate
    x_g = Conv1D(filters=n_filters,
                 kernel_size=filter_width,
                 padding='causal',
                 dilation_rate=dilation_rate)(x)

    # combine filter and gating branches
    z = Multiply()([Activation('tanh')(x_f),
                    Activation('sigmoid')(x_g)])

    # postprocessing - equivalent to time-distributed dense
    z = Conv1D(16, 1, padding='same', activation='relu')(z)

    # residual connection
    x = Add()([x, z])

    # collect skip connections
    skips.append(z)

# add all skip connection outputs
out = Activation('relu')(Add()(skips))

# final time-distributed dense layers
out = Conv1D(128, 1, padding='same')(out)
out = Activation('relu')(out)
out = Dropout(.2)(out)
out = Conv1D(1, 1, padding='same')(out)


# extract training target at end
def slice(x, seq_length):
    return x[:, -seq_length:, :]


pred_seq_train = Lambda(slice, arguments={'seq_length': 66})(out)

model = Model(history_seq, pred_seq_train)
model.compile(Adam(), loss='mean_absolute_error')

first_n_samples = 8
batch_size = 2 ** 10
epochs = 100

# sample of series from train_enc_start to train_enc_end
encoder_input_data = get_time_block_series(series_array, date_to_index,
                                           train_enc_start, train_enc_end)[:first_n_samples]

encoder_input_data, encode_series_mean = transform_series_encode(encoder_input_data)

# sample of series from train_pred_start to train_pred_end
decoder_target_data = get_time_block_series(series_array, date_to_index,
                                            train_pred_start, train_pred_end)[:first_n_samples]

decoder_target_data = transform_series_decode(decoder_target_data, encode_series_mean)

# we append a lagged history of the target series to the input data,
# so that we can train with teacher forcing
lagged_target_history = decoder_target_data[:, :-1, :1]
encoder_input_data = np.concatenate([encoder_input_data, lagged_target_history], axis=1)

### TRAINING
model.compile(Adam(), loss='mean_absolute_error')

history = model.fit(encoder_input_data, decoder_target_data,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.20)

plt.plot(np.exp(history.history['loss']))
plt.plot(np.exp(history.history['val_loss']))

plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error Loss')
plt.title('Loss Over Time')
plt.legend(['Train', 'Validation'])

plt.show()
