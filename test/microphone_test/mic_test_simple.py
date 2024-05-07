#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np
import time
import pyaudio
import wave
import zipfile
import wavio
import random
import asyncio
import IPython
from queue import Queue
from threading import Thread
np.set_printoptions(threshold=sys.maxsize)


# In[2]:


import sounddevice as sd
import soundfile as sf


# In[3]:


import tensorflow as tf


# In[4]:


sys.path.append("../ipynb_files/")
sys.path.append("../")


# In[5]:


from common import utils as U
from TestSharedLib.bytes import to_bytes, from_bytes, byte_conversion_tests, load_data, load_raw, save_raw, save_scores
from TestSharedLib.constants import quant_support, crops, feature_count


# In[6]:


from datetime import datetime
def genDataTimeStr():
    return datetime.today().strftime('%Y-%m-%d %H:%M:%S').replace('-',"").replace(' ',"").replace(':',"");


# In[7]:


# !pip install soundfile


# In[8]:


mask8 = 0x4000 # >> 8 : 16384
mask7 = 0x2000 # >> 7 :  8192
mask6 = 0x1000 # >> 6 :  4096
mask5 = 0x0800 # >> 5 :  2048
mask4 = 0x0400 # >> 4 :  1024
mask3 = 0x0200 # >> 3 :   512
mask2 = 0x0100 # >> 2 :   256
mask1 = 0x0080 # >> 1 :   128
mask0 = 0x0040 # >> 0 :    64 below the value, drop the value
def maskOP(x):
    x = np.int16(x)
    # print(f"begin:x:{x}")
    if (mask8&x):
        return x >> 8
    elif (mask7&x):
        return x >> 7
    elif (mask6&x):
        return x >> 6
    elif (mask5&x):
        return x >> 5
    elif (mask4&x):
        return x >> 4
    elif (mask3&x):
        return x >> 3
    elif (mask2&x):
        return x >> 2
    elif (mask1&x):
        return x >> 1
    elif (mask0&x):
        return x
    else:
        return 0;


# In[9]:


def quantize_int8_2(x, axis):
    # len_of_x = len(x[0][0][0])
    len_of_x = len(x[0][0])
    # print(f"len_of_x:{len_of_x}")
    for i in range(len_of_x):
        nflag = 2; #positive
        # print("{}:{}".format(i,x[0][0][i]))
        tmp_x = x[0][0][i]
        if tmp_x < 0:
            tmp_x = np.abs(tmp_x)
            nflag = 1
        tmp_x = maskOP(tmp_x)
        if(nflag==1):
            tmp_x = -1 * (tmp_x)
        # print("{}:{}".format(i,x[0][0][i]))
        # print("*********************************")
        x[0][0][i] = tmp_x
    return np.rint(x).astype(np.int8)


# In[10]:


def quantize_int8(x, axis):
  '''Quantization into int8_t precision, operating on x along axis'''
  scaling_factor_shape = tuple(np.append([len(x)],np.ones(x.ndim - 1, dtype = int)))
  epsilon = 0.000000001
  x_scaling_factor = (np.max(np.abs(x), axis) / 128) + epsilon
  x_scaling_factor = x_scaling_factor.reshape(scaling_factor_shape)
  # x_zero_offset = -0.5 #-0.25 #-0.25
  result = (x / x_scaling_factor) #+ x_zero_offset
  return np.rint(result).astype(np.int8)


# In[11]:


tflite_quant_model_path = "../../trained_models/step_6_QAT_and_Convert2TFLite/final_qat_model_lr0.01_testacc90.68_202405062253/qat_model_valacc92.7_tracc_82.9_prunInfo_0.8_0.85_ds_ver4home_20240506220725.tflite";
quanted_interpreter = tf.lite.Interpreter(model_path=tflite_quant_model_path)
print("model loaded....")


# In[12]:


input_details = quanted_interpreter.get_input_details()
output_details = quanted_interpreter.get_output_details()

print("== Input details ==")
print("name:", input_details[0]['name'])
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])

print("\n== Output details ==")
print("name:", output_details[0]['name'])
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])
#allocate tensor for testing
quanted_interpreter.allocate_tensors();


# ### sound preprocessing codes

# In[13]:


_inputLen = 30225
_nCrops = 2
def preprocess_setup():
    funcs = []
    funcs += [U.padding( _inputLen// 2),
              U.normalize(32768.0),
              U.multi_crop(_inputLen, _nCrops)]
              # U.single_crop(_inputLen)]
              # 

    return funcs

def preprocess_debug():
    debug_funcs = []
    debug_funcs += [U.padding( _inputLen// 2),
              # U.normalize(32768.0),]
              U.multi_crop(_inputLen, _nCrops)]
              # U.single_crop(_inputLen)]
              # 

    return debug_funcs


def preprocess(sound, funcs):
    for f in funcs:
        sound = f(sound)
    return sound;
    

def padding(pad):
    def f(sound):
        return np.pad(sound, pad, 'constant')

    return f
    

# def random_crop(size):
#     def f(sound):
#         org_size = len(sound)
#         start = random.randint(0, org_size - size)
#         return sound[start: start + size]

#     return f



# In[14]:


# _funcs = preprocess_debug()
_funcs = preprocess_setup()


# In[15]:


def doSoundClassification(input_wav=None, lblidx=None, channelIdx=0):
    sound = wavio.read(input_wav).data.T[0]
    start = sound.nonzero()[0].min();
    end = sound.nonzero()[0].max();
    sound = sound[start: end + 1];
    print(f"get sound signal from {start} to {end}");
    # if len(sound)> 220500:
    #     sound = sound[:220500]
    if len(sound)> 30225:
        sound = sound[:30226]
    # sound = np.int16(preprocess(sound, _funcs));
    sound = preprocess(sound, _funcs)
    # print(f"sound[{channelIdx}]:{sound[channelIdx]}");
    # print(f"sound[channelIdx] length is {len(sound[channelIdx])}")
    # label = label;
    s_test = np.expand_dims(sound[channelIdx], axis=0)
    s_test = np.expand_dims(s_test, axis=1);
    s_test = np.expand_dims(s_test, axis=3);
    # print(f"len of s_test:{len(s_test)}, shape of s_test:{s_test.shape}")
    s_test = quantize_int8_2(s_test,axis=-2)
    quanted_interpreter.set_tensor(input_details[0]['index'], s_test);
    quanted_interpreter.invoke()
    pred = quanted_interpreter.get_tensor(output_details[0]['index'])
    # print(f"Prediction result shape:{pred.shape}\n");
    print(f"Prediction result: {pred}, and true label idx: {lblidx}")
    # print(f"channel of inpu_wav:{len(sound)}");


# In[16]:


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1 #if sys.platform == 'darwin' else 2;
RATE = 20000
RECORD_SECONDS = 1.5
SAMPLE_SIZE = 2
FRAMES_PER_BUFFER = 64
STEAM_SAVE_DIR="./mic_record_savedir/stream_save/{}"


# In[17]:


def rcord_sound_by_soundfile(rec_druation = 2):
    print(f"進行{rec_druation}秒聲音錄製");
    WAVE_OUTPUT_FILENAME = "mic_test_sound_by_soundfile_{}.wav".format(genDataTimeStr());
    samplerate = 20000;#44100  # Hertz
    duration = rec_druation  # seconds
    channels = 1;
    test_wav = "./mic_record_savedir/{}".format(WAVE_OUTPUT_FILENAME);
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
                    channels=channels, blocking=True)
    sf.write(test_wav, mydata, samplerate);
    return test_wav;


# In[18]:


play_wav1 = None;
play_wav1 =  rcord_sound_by_soundfile(2);
print(f"living recording is {play_wav1}");
IPython.display.Audio(play_wav1)


# In[19]:


# doSoundClassification(input_wav=play_wav1,lblidx=1,channelIdx=1);


# In[21]:


test_wav2 = "./wail_280171.wav"
val_wav = "../../datasets/CurrentUse/wav_files/Single_Fold/val/positive/moaning_56/man_moaning_35_1_val_56.wav"
living_record_wav = "./mic_record_savedir/mic_test_sound_by_soundfile_20240507171455.wav"
print("test from live recording wav without modification");
doSoundClassification(input_wav=living_record_wav,lblidx=1,channelIdx=1)
doSoundClassification(input_wav=val_wav,lblidx=1,channelIdx=1)
doSoundClassification(input_wav=test_wav2,lblidx=1,channelIdx=0)


# In[37]:


streamTestMain()


# In[ ]:





# In[ ]:





# In[22]:


# p = pyaudio.PyAudio();
# #declare PyAudio stream
# stream = p.open(
# format=FORMAT,
# channels=CHANNELS,
# rate=RATE,
# input=True,
# frames_per_buffer=FRAMES_PER_BUFFER,)

# def closePyAudio():
#     stream.close();
#     p.termnate();


# In[2]:


# flag = False;
# frames1 = None;
# frames2 = None;
# async def asyncReceiveAndCallDoClassification():
#     async def getSignal():
#         global frames1, frames2, flag;
#         frames1 = [];
#         print("run in getSignal");
#         while True:
#             data = stream.read(FRAMES_PER_BUFFER);
#             frames1.append(data)
#             if len(frames1) >= (RATE * RECORD_SECONDS) / FRAMES_PER_BUFFER:
#                 print("run in getSignal if");
#                 flag = True
#                 frames2 = frame1.copy();
#                 frames1 = [];
#             await asyncio.sleep(0.01)
#         return True
#     async def consumeSignal():
#         global frames1, frames2, flag;
#         print("run in consumeSignal")
#         while True:
#             if flag == True:
#                 print("run in consumeSignal flag=true")
#                 doSoundClassificationbUsingStreaming(input_data=frames2);
#                 flag = False;
#             return asyncio.gather(getSignal(), consumeSignal())


# In[3]:


# await asyncReceiveAndCallDoClassification()


# In[2]:


# DURATION = 2;  # seconds

# def callback(in_data, frame_count, time_info, status):
#     print("data received.....");
#     #return (in_data, pyaudio.paContinue)


# stream = p.open(format=p.get_format_from_width(2),
#                 channels=1 if sys.platform == 'darwin' else 2,
#                 rate=20000,
#                 input=True,
#                 output=True,
#                 stream_callback=callback)

# start = time.time()
# while stream.is_active() and (time.time() - start) < DURATION:
#     pass
#     # time.sleep(1)

# stream.close()
# p.terminate()


# In[ ]:




