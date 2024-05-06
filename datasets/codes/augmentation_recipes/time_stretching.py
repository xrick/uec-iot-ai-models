#Time-stretching the wave
'''
Permissible factor values = 0 < x < 1.0
'''

factor = 0.4
wav_time_stch = librosa.effects.time_stretch(wav,factor)
plot_spec(data=wav_time_stch,sr=sr,title=f'Stretching the time by {factor}',fpath=file_path)
ipd.Audio(wav_time_stch,rate=sr)
# librosa.output.write_wav('./time_stech.wav',wav_time_stch,sr)