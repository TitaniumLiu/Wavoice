from pydub import AudioSegment
import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import scipy.io.wavfile as wav
from python_speech_features import logfbank
from functions import traverse, wav2logfbank, char_mapping
import argparse

parser = argparse.ArgumentParser(description="Librispeech preprocess.")

parser.add_argument(
    "--root",
    metavar="root",
    type = str,
    required=False,
    default="/data/home/Tiantian_Liu/data/mmWavoice_dataset/SNR4db/",
    help= "Absolute file path to mmWavioceDataset.",
)

parser.add_argument(
    "--n_jobs",
    dest="n_jobs",
    action="store",
    default=-2,
    help="number of cpu available for preprocessing. \n -1: use all cpu, -2: use all  cpu but one",
)

parser.add_argument(
    "--n_filters",
    dest="n_filters",
    action="store",
    default=40,
    help="number of flters for fbank. (Default: 40)",
)
parser.add_argument(
    "--win_size",
    dest="win_size",
    action="store",
    default=0.025,
    help="Window size during feature extraction (Default : 0.025 [25ms])",
)
parser.add_argument(
    "--norm_x",
    dest= "norm_x",
    action="store",
    default=False,
    help = "Normalize features s.t. mean = 0 ,std=1",
)


def main(args):
    root = args.root
    target_path = root + "/processed/"
    trainVoice_path = ["train_voice/"]
    trainmmWave_path = ["train_mmwave/"]
    devVoice_path = ["test_voice/"]
    devmmWave_path = ["test_mmwave/"]
    n_jobs = args.n_jobs
    n_filters = args.n_filters
    win_size = args.win_size
    norm_x = args.norm_x

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    print("-------------Processing Datasets--------------")
    print("Training Voice sets :", trainVoice_path)
    print("Training mmWave sets", trainmmWave_path)
    print("Validation Voice sets:", devVoice_path)
    print("Validation mmWave sets", devmmWave_path)
    print("-----------------------------------------------")

    tr_voicefile_list = traverse(root, trainVoice_path, search_fix=".wav")
    tr_mmwavefile_list = traverse(root, trainmmWave_path, search_fix=".wav")

    dev_voicefile_list = traverse(root, devVoice_path, search_fix=".wav")
    dev_mmwavefile_list = traverse(root, devmmWave_path, search_fix=".wav")

    print("________________________________________________")
    print("Processing wav2logfbank...", flush=True)

    print("Training Voice", flush=True)
    results=Parallel(n_jobs=n_jobs,backend="threading")(
        delayed(wav2logfbank)(i, win_size, n_filters) for i in tqdm(tr_voicefile_list)
    )

    print("Training mmWave", flush=True)
    results=Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(wav2logfbank)(i, win_size, n_filters) for i in tqdm(tr_mmwavefile_list)
    )
    print("Validation Voice", flush=True)
    results=Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(wav2logfbank)(i, win_size, n_filters) for i in tqdm(dev_voicefile_list)
    )

    print("Validation mmWave", flush=True)
    results=Parallel(n_jobs=n_jobs,backend="threading")(
        delayed(wav2logfbank)(i, win_size, n_filters) for i in tqdm(dev_mmwavefile_list)
    )



    # log-mel fbank 2 feature
    print("-------------------------------------------------")
    print("Preparing Training Dataset...", flush=True)

    tr_voicefile_list = traverse(root, trainVoice_path, search_fix=".fb" + str(n_filters))
    tr_text = traverse(root, trainVoice_path, return_label=True)
    tr_mmwavefile_list = traverse(root, trainmmWave_path, search_fix=".fb"+str(n_filters))

    #Create char mapping
    char_map = char_mapping(tr_text, target_path)

    #text to index sequence
    tmp_list = []
    for text in tr_text:
        tmp = []
        for char in text:
            tmp.append(char_map[char])
        tmp_list.append(tmp)
    tr_text = tmp_list
    del tmp_list

    #write dataset

    file_name = "train.csv"
    print("Writing dataset to " + target_path + file_name + "...", flush=True)
    with open(target_path + file_name, "w") as f:
        f.write("idx, voice_input, mmwave_input, label\n")
        for i in range(len(tr_voicefile_list)):
            f.write(str(i) + ",")
            f.write(tr_voicefile_list[i] + ",")
            f.write(tr_mmwavefile_list[i] + ",") # train mmwave path
            for char in tr_text[i]:
                f.write(" " + str(char))
            f.write("\n")
    print()
    print("Preparing Validation Dataset...", flush=True)
    dev_voicefile_list = traverse(root, devVoice_path, search_fix=".fb"+str(n_filters))
    dev_mmwavefile_list = traverse(root, devmmWave_path, search_fix=".fb" + str(n_filters))
    dev_text = traverse(root, devVoice_path, return_label=True)
    tmp_list = []
    for text in dev_text:
        tmp = []
        for char in text:
            tmp.append(char_map[char])
        tmp_list.append(tmp)
    dev_text = tmp_list
    del tmp_list

    #write dataset
    file_name = "dev.csv"
    print("Writing dataset to " + target_path+file_name+"...", flush=True)

    with open(target_path + file_name, "w") as f:
        f.write("idx, voice_input, mmwave_input, label\n")
        for i in range(len(dev_voicefile_list)):
            f.write(str(i) + ",")
            f.write(dev_voicefile_list[i] + ",")
            f.write(dev_mmwavefile_list[i] + ",")
            for char in dev_text[i]:
                f.write(" " + str(char))
            f.write("\n")
    print()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)




