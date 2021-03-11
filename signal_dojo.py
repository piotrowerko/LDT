import matplotlib.pyplot as plt
from scipy.signal import hilbert 
from scipy.optimize import least_squares

from signal.signal import Signal

def main():
    # define object
    bridge_radar_dyn = Signal('\nzestaw1\n')
    # print object name
    print(bridge_radar_dyn)
    #import data
    bridge_radar_dyn.load_data('./zestaw1.txt')
   
    #check key parameters of sampling features of the input signal from in-situ device
    print(f'mean, median, and standard deviation of input sampling rate: \n{bridge_radar_dyn.compute_sampling_spacing()}\n')
    #check key parameters of period features of the structural reposnse
    print(f'mean, median, and standard deviation of signal main period: \n{bridge_radar_dyn.compute_period()}\n')

    #return LSF object cost function
    #print(f'LSF object cost function: {bridge_radar_dyn.lsq_resutls().cost}\n')
    #return LSF object object's solve vector
    bridge_radar_dyn.fourier_trans()  # dodanie wyznaczenia częstotliwości głównej na potrzeby progu filtra
    bridge_radar_dyn.lowpass_filter(1.5*bridge_radar_dyn.main_freq)  # dodanie filtracji sygnału (dolnoprzepustowy)
    #bridge_radar_dyn.bandpass_filter(0.1*bridge_radar_dyn.main_freq, 1.5*bridge_radar_dyn.main_freq)  # dodanie filtracji sygnału (pasmoprzepustowy)
    bridge_radar_dyn.compute_envelope()
    print(f'LSF objects solve vector: {bridge_radar_dyn.lsq_resutls().x}\n')
    # return Soft_L1 fit object
    #print(f'Soft_L1 fit object: {bridge_radar_dyn.softL1_results()}\n')
    # return Soft_L1 object's solve vector
    print(f'Soft_L1 fit object: {bridge_radar_dyn.softL1_results().x}\n')
    # return Huber fit object's solve vector
    print(f'Huber fit objects solve vector: {bridge_radar_dyn.huber_results().x}')

    #plotting
    fig, ax = plt.subplots()
    ax.plot(bridge_radar_dyn.time, bridge_radar_dyn.Amplitude_, label='signal')
    ax.plot(bridge_radar_dyn.time, bridge_radar_dyn.Amplitude, label='filtered_signal')
    ax.plot(bridge_radar_dyn.time_cutted, bridge_radar_dyn.amplitude_envelope_cutted, label='envelope')
    LSF_fun_points = bridge_radar_dyn.fun(bridge_radar_dyn.lsq_resutls().x, bridge_radar_dyn.time_cutted, 0)
    ax.plot(bridge_radar_dyn.time_cutted, LSF_fun_points, label='linear loss')
    softL1_fun_points = bridge_radar_dyn.fun(bridge_radar_dyn.softL1_results().x, bridge_radar_dyn.time_cutted, 0)
    #softL1_fun_points = [bridge_radar_dyn.fun(bridge_radar_dyn.softL1_results().x, i, 0) for i in bridge_radar_dyn.time] # iteracja listą - 20 sek. wolniej
    ax.plot(bridge_radar_dyn.time_cutted, softL1_fun_points, label='softL1 loss')
    huber_fun_points = bridge_radar_dyn.fun(bridge_radar_dyn.huber_results().x, bridge_radar_dyn.time_cutted, 0)
    #huber_fun_points = [bridge_radar_dyn.fun(bridge_radar_dyn.huber_results().x, i, 0) for i in bridge_radar_dyn.time] # iteracja listą - 20 sek. wolniej
    ax.plot(bridge_radar_dyn.time_cutted, huber_fun_points, label='huber loss')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Rel. vert. displacement [mm]')
    plt.legend()
    plt.show()
    
    # Frequency domain representation - plot
    fig2, ax2 = plt.subplots()  # przywrócenie wykresu fft
    ax2.set_title('Fourier transform depicting the frequency components')
    ax2.plot(bridge_radar_dyn.frequecies_cutted, abs(bridge_radar_dyn.fourierTransform_cutted))
    plt.xticks([bridge_radar_dyn.main_freq])
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Amplitude')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
    