import re
import os
import subprocess
def getFileList(srcDir,regex='.*\.wav'):
    # example: regex = '.*\.mp3'
    results = os.listdir(srcDir)
    out_files = []
    cnt_files = 0
    for file in results:
        # if not file.startswith('.') and os.path.isdir(os.path.join(srcDir, file)):
        if not file.startswith('.') and os.path.isdir(os.path.join(srcDir, file)):
            out_files += getFileList(os.path.join(srcDir, file))
        elif re.match(regex, file,  re.I):  # file.startswith(startExtension) or file.endswith(".txt") or file.endswith(endExtension):
            out_files.append(os.path.join(srcDir, file))
            cnt_files = cnt_files + 1
    return out_files


def Convert(src_wav, dest_wav, sr):
    subprocess.call('ffmpeg -i {} -ac 1 -ar {} -loglevel error -y {}'.format(
            src_wav, sr, dest_wav), shell=True);

def getFolderList(rootDir=None, recursive=False):
    if not recursive:
        return next(os.walk(rootDir));
    else:
        return [x[0] for x in os.walk(rootDir)]