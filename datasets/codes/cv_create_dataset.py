import sys
import os
import subprocess
import glob
import numpy as np
import wavio

sys.path.append('../../');
from Libs.SharedLibs import getFileList;

def main(dataset_dir=None, fold_dirs=None, output_path=None, if_convert_sr=None, sr_list=[44100, 20000, 16000], folds=5):
    # mainDir = os.getcwd();
    # esc50_path = os.path.join(mainDir, 'datasets/esc50');

    # if not os.path.exists:
    #     os.mkdir(esc50_path)

    # Convert sampling rate
    if if_convert_sr:
        for sr in sr_list:
            pass
            # if sr == 44100:
            #     continue
            # else:

            #                os.path.join(esc50_path, 'wav{}'.format(sr // 1000)),
            #                sr);

    # Create npz files
    # for sr in sr_list:
    #    if sr == 44100:
    #        src_path = os.path.join(esc50_path, 'ESC-50-master', 'audio');
    #    else:
    #        src_path = os.path.join(esc50_path, 'wav{}'.format(sr // 1000));
    create_dataset(dataset_dir,fold_dirs, output_path, folds);


def convert_sr(src_path, dst_path, sr):
    print('* {} -> {}'.format(src_path, dst_path))
    if not os.path.exists(dst_path):
        os.mkdir(dst_path);
    for src_file in sorted(glob.glob(os.path.join(src_path, '*.wav'))):
        dst_file = src_file.replace(src_path, dst_path);
        subprocess.call('ffmpeg -i {} -ac 1 -ar {} -loglevel error -y {}'.format(
            src_file, sr, dst_file), shell=True);

def create_dataset(src_path=None, fold_dirs=None, output_path=None, folds=5):
    print('* {} -> {}'.format(src_path, output_path))
    train_dataset = {};

    for fold in range(1, folds+1):
        train_dataset['fold{}'.format(fold)] = {}
        train_sounds = []
        train_labels = []

        for wav_file in sorted(glob.glob(os.path.join(src_path, '{}-*.wav'.format(fold)))):
            sound = wavio.read(wav_file).data.T[0]
            start = sound.nonzero()[0].min()
            end = sound.nonzero()[0].max()
            sound = sound[start: end + 1]  # Remove silent sections
            label = int(os.path.splitext(wav_file)[0].split('-')[-1])
            train_sounds.append(sound)
            train_labels.append(label)

        train_dataset['fold{}'.format(fold)]['sounds'] = train_sounds
        train_dataset['fold{}'.format(fold)]['labels'] = train_labels

    np.savez(output_path, **train_dataset)

def create_wav_list(wav_src_path)->list:
    wav_list = [];
    return wav_list;


if __name__ == '__main__':
    main()
