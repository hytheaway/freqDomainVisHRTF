import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# uncomment if you're timing
# import time
#
# only uncomment if using numba
# import numba

hrir_file_path = "000e030a.wav"  # mit kemar as an example
hrir_data, hrir_sr = sf.read(hrir_file_path)


# numba actually slows down this computation. (0.117s without; 0.779s with on my base m2 macbook air)
# only uncomment if using numba
# @numba.njit
def hrtf_calculation():
    nfft = len(hrir_data) * 8
    hrtf_data = np.fft.fft(hrir_data, n=nfft, axis=0)
    hrtf_mag = (2 / nfft) * np.abs(hrtf_data[0 : int(len(hrtf_data) / 2) + 1, :])
    hrtf_mag_db = 20 * np.log10(hrtf_mag)

    f_axis = np.linspace(0, hrir_sr / 2, len(hrtf_mag_db))
    return f_axis, hrtf_mag_db


def plotting_function(in_f_axis, in_hrtf_mag_db):
    plt.figure(num=("Frequency domain representation for given HRIR"))
    plt.semilogx(in_f_axis, in_hrtf_mag_db)
    plt.grid()
    plt.grid(which="minor", color="0.9")
    plt.title("HRTF for given HRIR")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend(["Left", "Right"])

    # uncomment these if you're timing
    # plt.show(block=False)
    # plt.close()

    # comment this out if you're timing
    plt.show()


def main_function():
    f_axis, hrtf_mag_db = hrtf_calculation()
    plotting_function(f_axis, hrtf_mag_db)


if __name__ == "__main__":
    # uncomment these commented if you're timing
    # start_time = time.time()
    main_function()
    # print("---- %s seconds ----" % (time.time() - start_time))
