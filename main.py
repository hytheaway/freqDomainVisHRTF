import numpy as np
import matplotlib.pyplot as plt
import time
import soundfile as sf

hrir_file_path = "000e030a.wav"  # mit kemar as a sample
hrir_data, hrir_sr = sf.read(hrir_file_path)


def main_function():
    start_time = time.time()
    nfft = len(hrir_data) * 8
    hrtf_data = np.fft.fft(hrir_data, n=nfft, axis=0)
    hrtf_mag = (2 / nfft) * np.abs(hrtf_data[0 : int(len(hrtf_data) / 2) + 1, :])
    hrtf_mag_db = 20 * np.log10(hrtf_mag)

    f_axis = np.linspace(0, hrir_sr / 2, len(hrtf_mag_db))
    plt.figure(num=("Frequency domain representation for given HRIR"))
    plt.semilogx(f_axis, hrtf_mag_db)
    plt.grid()
    plt.grid(which="minor", color="0.9")
    plt.title("HRTF for given HRIR")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend(["Left", "Right"])
    print("---- %s seconds ----" % (time.time() - start_time))
    plt.show()


if __name__ == "__main__":
    main_function()
