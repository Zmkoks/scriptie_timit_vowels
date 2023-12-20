## Input: wav files, PHN files
## output: mfcc of points in WAV files, connecting with PHN files (annotated = variable)

import librosa
import numpy
import tqdm
import os


COUNTER = 0

def parse_id_name_phone_array(id_name, phone, i):
    new_array = [
        id_name[2:3],
        id_name[3:4],
        id_name[4:8],
        id_name[8:10],
        id_name[10:],
        phone,
        str(i)]
    return new_array


def read_phn_file(phn_file, id_name, data_x, hoplength, groups):
    length_data_x = data_x.shape[0]
    y_data = numpy.full((length_data_x, 7), "", dtype="U6")
    with open(phn_file, "r") as phn:
        lines = phn.readlines()
    cnt = 0
    for line in lines:
        start, end, phone = line.split(' ')
        phone = phone.strip()
        if phone in groups:
            phone = groups[phone]
        end = int(end) // hoplength
        start = int(start) // hoplength

        for i in range(start, end):
            list_y = parse_id_name_phone_array(id_name, phone, i)
            arr = numpy.array(list_y)
            y_data[cnt] = arr
            cnt += 1
            if cnt > length_data_x:
                cnt -= 1
                global COUNTER
                print("somethings getting overwritten: {}".format(COUNTER))
                COUNTER += 1
    return y_data


def make_mfcc(audio_wav_file, hoplength, framelength, n_mels):
    x, sr = librosa.load(audio_wav_file, sr=None)
    mfcc = librosa.feature.mfcc(x, sr=sr, hop_length=hoplength, n_mfcc=framelength, n_mels=n_mels)  # slices van 20 ms
    return (mfcc, sr)


def create_data(audio_wav_file, hoplength, framelength, n_mels, groups):
    id_name = os.path.splitext(os.path.basename(audio_wav_file))[0]
    file_name = os.path.splitext(audio_wav_file)[0]
    phn_file = file_name + ".PHN"
    print(id_name, end='\r')
    mfcc_list, sample_rate = make_mfcc(audio_wav_file, hoplength, framelength,n_mels)
    data_x = numpy.moveaxis(mfcc_list, 1, 0)
    ydata_array = read_phn_file(phn_file, id_name, data_x, hoplength, groups)
    data_all = numpy.column_stack((ydata_array, data_x))
    return (data_all)


def loop_through_folder(name, path, hoplength, framelength, n_mels, groups):
    for i in ["TEST","TRAIN"]:
        new_path = os.path.join(path,i)

        data_str = open(name+"_"+i+".txt", "a")
        for file in tqdm.tqdm((os.listdir(new_path))):
            if file.endswith(".WAV"):
                new_data = create_data(os.path.join(new_path, file), hoplength, framelength, n_mels, groups)
                numpy.savetxt(data_str, new_data, delimiter=",", fmt="%s")

        data_str.close()

def run_main(name, path, hoplength = 160, framelength = 400, n_mels = 13, groups = None):
    if groups is None:
        groups = {}
    if not (os.path.exists(name+"_TRAIN.txt") and os.path.exists(name+"_TEST.txt")):
        loop_through_folder(name, path, hoplength, framelength, n_mels, groups)
