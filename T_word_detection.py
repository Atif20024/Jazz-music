
# coding: utf-8

# In[ ]:



import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from td_utils import *
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#visualizing an audio
x = graph_spectrogram("audio_examples/example_train.wav")


# In[ ]:


_, data = wavfile.read("audio_examples/example_train.wav")


# In[ ]:


Tx = 5511
n_freq = 101


# In[ ]:


Ty = 1375


# In[ ]:


activates, negatives, backgrounds = load_raw_audio()

print("background len: " + str(len(backgrounds[0])))    # Should be 10,000, since it is a 10 sec clip
print("activate[0] len: " + str(len(activates[0])))     # Maybe around 1000, since an "activate" audio clip is usually around 1 sec (but varies a lot)
print("activate[1] len: " + str(len(activates[1])))


# In[ ]:



def get_random_time_segment(segment_ms):
    segment_start = np.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background 
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)


# In[ ]:


def is_overlapping(segment_time, previous_segments):
    segment_start, segment_end = segment_time
    overlap = False
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True
    return overlap


# In[ ]:


overlap1 = is_overlapping((950, 1430), [(2000, 2550), (260, 949)])
overlap2 = is_overlapping((2305, 2950), [(824, 1532), (1900, 2305), (3424, 3656)])
print("Overlap 1 = ", overlap1)
print("Overlap 2 = ", overlap2)


# In[ ]:



def insert_audio_clip(background, audio_clip, previous_segments):
    segment_ms = len(audio_clip)
    
    segment_time = get_random_time_segment(segment_ms)
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    previous_segments.append(segment_time)
    new_background = background.overlay(audio_clip, position = segment_time[0])
    
    return new_background, segment_time


# In[ ]:


IPython.display.Audio("audio_examples/insert_reference.wav")


# In[ ]:



def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 followinf labels should be ones.
    
    
    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms
    
    Returns:
    y -- updated labels
    """
    
    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    
    # Add 1 to the correct index in the background label (y)
    ### START CODE HERE ### (≈ 3 lines)
    for i in range(segment_end_y + 1, segment_end_y + 51):
        if i < Ty:
            y[0, i] = 1
    ### END CODE HERE ###
    
    return y


# In[ ]:


def create_training_example(background, activates, negatives):
    np.random.seed(18)
    background = background - 20
    y = np.zeros((1, Ty))
    previous_segments = []

    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    
    for random_activate in random_activates:
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end_ms=segment_end)
    
    
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    for random_negative in random_negatives:
        background, _ = insert_audio_clip(background, random_negative, previous_segments)
    
    background = match_target_amplitude(background, -20.0)
    
    file_handle = background.export("train" + ".wav", format="wav")
    print("File (train.wav) was saved in your directory.")
    x = graph_spectrogram("train.wav")
    
    return x, y


# In[ ]:


x, y = create_training_example(backgrounds[0], activates, negatives)


# In[ ]:


IPython.display.Audio("train.wav")

IPython.display.Audio("audio_examples/train_reference.wav")
plt.plot(y[0])


# In[ ]:



X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")


# In[ ]:


from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam


# In[ ]:


def model(input_shape):
    
    X_input = Input(shape = input_shape)
    
    X = Conv1D(196, kernel_size=15, strides=4)(X_input)       # CONV1D
    X = BatchNormalization()(X)                               # Batch normalization
    X = Activation('relu')(X)                                 # ReLu activation
    X = Dropout(0.8)(X)                                       # dropout (use 0.8)

    X = GRU(units = 128, return_sequences = True)(X)          # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                       # dropout (use 0.8)
    X = BatchNormalization()(X)                               # Batch normalization
    
    X = GRU(units = 128, return_sequences = True)(X)          # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                       # dropout (use 0.8)
    X = BatchNormalization()(X)                               # Batch normalization
    X = Dropout(0.8)(X)                                       # dropout (use 0.8)
    
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)  # time distributed  (sigmoid)

    model = Model(inputs = X_input, outputs = X)
    
    return model


# In[ ]:


model = model(input_shape = (Tx, n_freq))


# In[ ]:


model.summary()


# In[ ]:


model = load_model('./models/tr_model.h5')


# In[ ]:



opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])


# In[ ]:


model.fit(X, Y, batch_size = 5, epochs=1)


# In[ ]:


loss, acc = model.evaluate(X_dev, Y_dev)
print("Dev set accuracy = ", acc)


# In[ ]:


def detect_triggerword(filename):
    plt.subplot(2, 1, 1)

    x = graph_spectrogram(filename)
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    return predictions


# In[ ]:



chime_file = "audio_examples/chime.wav"
def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    consecutive_timesteps = 0
    for i in range(Ty):
        consecutive_timesteps += 1
        if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
            consecutive_timesteps = 0
        
    audio_clip.export("chime_output.wav", format='wav')


# In[ ]:



IPython.display.Audio("./raw_data/dev/1.wav")


# In[ ]:


filename = "./raw_data/dev/1.wav"
prediction = detect_triggerword(filename)
chime_on_activate(filename, prediction, 0.5)
IPython.display.Audio("./chime_output.wav")


# In[ ]:


#check with your own voice
filename  = "./raw_data/dev/2.wav"
prediction = detect_triggerword(filename)
chime_on_activate(filename, prediction, 0.5)
IPython.display.Audio("./chime_output.wav")


# In[ ]:


def preprocess_audio(filename):
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    segment = padding.overlay(segment)
    segment = segment.set_frame_rate(44100)
    segment.export(filename, format='wav')


# In[ ]:


your_filename = "audio_examples/my_audio.wav"


# In[ ]:


# listen to the audio you uploaded
preprocess_audio(your_filename)
IPython.display.Audio(your_filename) 


# In[ ]:


chime_threshold = 0.5
prediction = detect_triggerword(your_filename)
chime_on_activate(your_filename, prediction, chime_threshold)
IPython.display.Audio("./chime_output.wav")

