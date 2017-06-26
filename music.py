import os
import wave
import pyaudio
import numpy as np
from pydub import AudioSegment
from glob import glob
from lstm_music import model_music
# from sklearn.cluster import MiniBatchKMeans


music_dir = "D:/Music"
os.chdir(music_dir)


def read_wav(path):
    # time=np.arange(0,nframes)*(1.0/framerate)
    f_ = wave.open(path, 'rb')
    params = f_.getparams()
    [nchannels, sampwidth, framerate, nframes] = params[:4]
    data = f_.readframes(nframes)
    if sampwidth <= 2:
        wave_data = np.fromstring(data, dtype=np.short)
    else:
        wave_data = np.fromstring(data, dtype=int)
    if nchannels == 1:
        return wave_data
    else:
        wave_data = np.reshape(wave_data,(-1,nchannels)).T
        return wave_data[0]


def write_wav(data,path,nchannels,sampwidth,framerate):
    f_ = wave.open(path, 'wb')
    f_.setnchannels(nchannels)
    f_.setsampwidth(sampwidth)
    f_.setframerate(framerate)
    f_.writeframes(data.tostring())
    f_.close()


def play_wav(path):
    f = wave.open(path, 'rb')
    params = f.getparams()
    print(params[:4])
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                    channels=f.getnchannels(),
                    rate=f.getframerate(),
                    output=True)
    data = f.readframes(params[3])
    #while len(data) != 0:
    stream.write(data)
    stream.stop_stream()
    stream.close()
    p.terminate()


def get_indmat(n,maxlen,step):
    xind = []
    yind=np.arange(maxlen,n-1,step)
    for i in yind:
        if len(xind) == 0:
            xind = np.arange(i-maxlen,i)
        else:
            xind = np.vstack([xind,np.arange(i-maxlen,i)])
    return xind,yind


def from_note(x_,labels,output_dim):
    x_mat = np.zeros((len(x_),output_dim))
    for i in range(len(x_)):
        x_mat[i, labels[x_[i]]] = 1
    return np.expand_dims(x_mat,axis=0)


def to_note(y_,labels):
    label = np.argmax(y_)
    note_ind = np.where(labels == label)[0]
    note_sel = np.random.choice(note_ind,1)[0]
    return note_sel


def generate_music(model_,labels,input_length,output_dim,vocab_num,wavMat):
    for j in range(10):
        music_ind = []
        start_index = np.random.randint(0, vocab_num-input_length-1)
        piece = np.arange(start_index, start_index+input_length)
        x_new = from_note(piece,labels,output_dim)
        for t in range(input_length):
            y_pred = model_.predict(x_new)[0]
            note = to_note(y_pred,labels)
            piece = np.append(piece[1:], note)
            #start_index += 1
            #piece = np.arange(start_index, start_index+input_length)
            x_new = from_note(piece,labels,output_dim)
            music_ind.append(note)
        music = wavMat[music_ind, :]
        music = np.ravel(music)
        write_wav(music, 'Music'+str(j)+'.wav', 1, 2, 44100)
        AudioSegment.from_wav('Music'+str(j)+'.wav').export('Music'+str(j)+'.mp3', format='mp3')


def play_all():
    filenames=['Music'+str(j)+'.mp3' for j in range(10)]
    song = []
    for mp3_file in filenames:
        print(mp3_file)
        if len(song) == 0:
            song = AudioSegment.from_mp3(mp3_file)
        else:
            song += AudioSegment.from_mp3(mp3_file)
    song.export('Music.mp3', format='mp3')
    AudioSegment.from_mp3('Music.mp3').export('Music.wav', format='wav')
    play_wav("Music.wav")


def main(timelen,mp3_file):
    wavMat = []
    indMat = []
    indY = []
    wavLen = [0]
    wavDict = {}
    FRAMERATE = 44100
    framerate = int(FRAMERATE*timelen)
    maxlen = int(30/timelen)
    step = int(1/timelen)

    AudioSegment.from_mp3(mp3_file).export(mp3_file[:-4]+'.wav', format='wav')
    for wav_file in glob('*.wav'):
        wav_data = read_wav(wav_file)
        wav_data = wav_data[FRAMERATE:-FRAMERATE*5]
        n = len(wav_data) - len(wav_data) % framerate
        wav_data = np.reshape(wav_data[:n], (-1,framerate))
        n_ = len(wav_data)
        indmat,indy = get_indmat(n_,maxlen,step)
        indmat += sum(wavLen)
        indy += sum(wavLen)
        wavLen.append(n_)
        wavDict[wav_file] = n_
        print(wav_file,n,n_)
        if len(wavMat) == 0:
            wavMat = wav_data
            indMat = indmat
            indY = indy
        else:
            wavMat = np.vstack([wavMat, wav_data])
            indMat = np.vstack([indMat, indmat])
            indY = np.hstack([indY, indy])
    vocab_num = sum(wavLen)
    print(wavDict)
    print(wavMat.shape)
    print(indMat.shape)

    # outdim = 100
    # clusters = MiniBatchKMeans(n_clusters=outdim,max_iter=300).fit(wavMat)
    # labels = clusters.labels_
    outdim = len(wavMat)
    labels = np.arange(0,len(wavMat))
    x = np.zeros((len(indMat),maxlen,outdim))
    y = np.zeros((len(indY),outdim))
    for k in range(len(indMat)):
        y[k, labels[indY[k]]] = 1
        for j in range(maxlen):
            x[k, j, labels[indMat[k,j]]] = 1
    pass
    model = model_music(x, y, 0.2)
    generate_music(model, labels, maxlen, outdim, vocab_num,wavMat)


if __name__ == "__main__":
    main(1, "天空之城.mp3")
    play_all()
