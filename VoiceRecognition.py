# first comment
import matplotlib.pyplot as plt
import numpy
from numpy import mean
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
import os
# Assign my audio path.
def get_filepaths(directory):
    file_paths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


full_file_paths = get_filepaths("/home/majutharan/Downloads/Emotion/Happy")

meanOfDecibel = []
meanOfAplitude = []

amplitudeData = "amplitudeDataForHappy.csv"
decibelData = "decibelDataForHappy.csv"

for f in full_file_paths:
    if f.endswith(".wav"):
        myAudio = f
        # Read file and get sampling frequency and sound object.
        samplingFreq, mySound = wavfile.read(myAudio)
        frequency, time, spectogram = signal.spectrogram(mySound, samplingFreq)

        # print sound object Ex: [1355, 955] (samples)
        print("my_sound:", mySound)

        # print sampling value(samples/sec)[Hz].
        print("sampling_freq:", samplingFreq)

        # Find the data type of sound file.
        # Ex: int 16, int 32
        mySoundDataType = mySound.dtype

        # print the value of sound bit.
        print('my_Sound_DataType:', mySoundDataType)

        print('Max value of sampling:', mySound.max())
        print('Min value of sampling:', mySound.min())
        print('mean of sound:', mean(mySound))

        # Convert the sound array to floating points [-1 to +1]
        mySound = mySound / 2 ** 15

        # Print the floating points.
        print('sound value in float:', mySound)

        # Check sound points. Its dual-channel or mono channel
        mySoundShape = mySound.shape

        # Analysis the how many samples are in sound object.
        samplePoints = float(mySound.shape[0])

        # Print the samples count.
        print('SamplePoints', samplePoints)
        print('sound shape', mySoundShape)

        # Find the signal duration = samples / (samples / time)
        signalDuration = mySound.shape[0] / samplingFreq

        # Print the duration of wave file.
        print('signal duration', signalDuration)

        # If two channel then get one channel

        print('mySound', mySound)

        mySoundOneChannel = mySound[:, 0]
        print('mySoundOneChannel:', mySoundOneChannel)

        # Arange the value of sample points within 1 to 169629 [space 1]
        timeArray = numpy.arange(0, samplePoints, 1)
        print('time array:', timeArray)

        # time array = samplePints array / (samples / t) => t
        timeArray = timeArray / samplingFreq
        print('time array new:', timeArray)

        # Change to s(seconds)  to ms(milli seconds)
        timeArray = timeArray * 1000

        # Print s(seconds) value to ms(milli seconds)
        print('time array * 1000 :', timeArray)
        meanOfAplitude.append(float(mean(mySoundOneChannel))*(-10000000000))
        print('mean of amplitude', meanOfAplitude)
        numpy.savetxt(amplitudeData, meanOfAplitude, fmt="%s")

        # # make graph to amplitude vs time
        # plt.plot(timeArray, mySoundOneChannel, color='G')
        # plt.ylabel('Amplitude')
        # plt.xlabel('Time')
        # plt.show()

        # calculate the sound length
        mySoundLength = len(mySound)
        print('my sound length is :', mySoundLength)

        # fast frequency transformation of sound clip
        # Analysis the sound clip via frequency.
        fftArray = fft(mySoundOneChannel)
        print('fast frequency transformation of mySoundOneChannel)', fftArray)

        numUniquePoints = int(numpy.ceil((mySoundLength + 1) // 2))
        print('numUniquePoints:', numUniquePoints)

        fftArray = fftArray[0:numUniquePoints]
        print('fastFrequencyTransformation:', fftArray)

        fftArray = abs(fftArray)
        print('fftArray:', fftArray)

        fftArray = fftArray / float(mySoundLength)
        print('fftArray:', fftArray)

        # change to positive numbers.
        fftArray = fftArray ** 2

        # Multiply by two
        # Odd NFFT excludes Nyquist point
        if mySoundLength % 2 > 0:  # we've got odd number of points in fft
            fftArray[1:len(fftArray)] = fftArray[1:len(fftArray)] * 2

        else:  # We've got even number of points in fft
            fftArray[1:len(fftArray) - 1] = fftArray[1:len(fftArray) - 1] * 2

        freqArray = numpy.arange(0, numUniquePoints, 1.0) * (samplingFreq / mySoundLength)


        #log algorithm

        log = 10 * numpy.log10(fftArray)
        print('log', log)
        print('mean', mean(log))

        # # Plot the frequency
        # plt.plot(freqArray / 1000, 10 * numpy.log10(fftArray), color='B')
        # plt.xlabel('Frequency (Khz)')
        # plt.ylabel('Power (dB)')
        # plt.show()

        # Get List of element in frequency array
        # print freqArray.dtype.type
        freqArrayLength = len(freqArray)
        print("freqArrayLength =", freqArrayLength)
        meanOfDecibel.append(mean(log)* (-1))
        numpy.savetxt(decibelData, meanOfDecibel, fmt="%s")

        # Print FFtarray information
        print("fftArray length =", len(fftArray))

        spectogram1D = spectogram[:, 0]
        spectogram1D = spectogram1D[:, 0]


        print('frequency is :', frequency)
        print('time is :', time)
        print('spectogram is :', spectogram1D)


print('meanOfAplitude', meanOfAplitude)
print ('meanOfDecibel', meanOfDecibel)








