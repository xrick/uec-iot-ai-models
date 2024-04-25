'''
Noise addition using normal distribution with mean = 0 and std =1

Permissible noise factor value = x > 0.004
'''
wav_n = wav + 0.009*np.random.normal(0,1,len(wav))
plot_spec(wav_n,sr,'Noise Added 0.005',file_path)
ipd.Audio(data=wav_n,rate=sr)
# librosa.output.write_wav('./noise_add.wav',wav_n,sr)