import matplotlib.pyplot as plt
from scipy.signal import hilbert 
from scipy.optimize import least_squares
import numpy as np

from signal.signal_in_min import MinuteSignal

def main():
    # define object
    bridge_poj_dyn = MinuteSignal('signal_wl_50')
    # print object name
    print(bridge_poj_dyn)
    #import data
    bridge_poj_dyn.load_data_in_minutes('wl_50.txt')
    #check key parameters of sampling features of the input signal from in-situ device
    print(f'mean, median, and standard deviation of input sampling rate: \n{bridge_poj_dyn.compute_sampling_spacing()}\n')
    #check key parameters of period features of the structural reposnse
    print(f'mean, median, and standard deviation of signal main period: \n{bridge_poj_dyn.compute_period()}\n')

    #return LSF object cost function
    #print(f'LSF object cost function: {bridge_poj_dyn.lsq_resutls().cost}\n')
    #return LSF object object's solve vector
    bridge_poj_dyn.fourier_trans().lowpass_filter(1.5*bridge_poj_dyn.main_freq).compute_envelope(0.90)
    print(f'LSF objects solve vector: {bridge_poj_dyn.lsq_resutls().x}\n')
    # return Soft_L1 fit object
    #print(f'Soft_L1 fit object: {bridge_poj_dyn.softL1_results()}\n')
    # return Soft_L1 object's solve vector
    print(f'Soft_L1 fit object: {bridge_poj_dyn.softL1_results().x}\n')
    # return Huber fit object's solve vector
    print(f'Huber fit objects solve vector: {bridge_poj_dyn.huber_results().x}')

    
    #plotting_robust
    fig0, ax0 = plt.subplots()
    fig0.set_size_inches(9, 7)
    ax0.plot(bridge_poj_dyn.time, bridge_poj_dyn.Amplitude, label='signal')
    ax0.plot(bridge_poj_dyn.time_cutted, bridge_poj_dyn.amplitude_envelope_cutted, label='envelope')
    huber_fun_points = bridge_poj_dyn.fun(bridge_poj_dyn.huber_results().x, bridge_poj_dyn.time_cutted, 0)
    plt.text(5.5, 0.20, r'$x = Ae^{-\beta t} + c$', {'color': 'C2'},fontsize=15)
    param = bridge_poj_dyn.huber_results().x
    plt.text(7.4, 0.15, r'$\beta$', {'color': 'C2'},fontsize=15)
    plt.text(5.5, 0.15, f'A: {np.round(param[0], 3)},   : {np.round(param[1], 3)}, c: {np.round(param[2], 3)}', {'color': 'C2'},fontsize=15)
    ax0.plot(bridge_poj_dyn.time_cutted, huber_fun_points, label='huber loss')
    ax0.tick_params(labelsize=15)
    ax0.set_xlabel('Time [s]', fontsize=15)
    ax0.xaxis.set_label_coords(1.06, -0.025)
    ax0.set_ylabel('Rel. vert. displacement [mm]', fontsize=15, rotation=90)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3, fontsize=15)
    plt.savefig('figure.jpeg', dpi=600)
    plt.show()
    
    #plotting
    fig, ax = plt.subplots()
    ax.plot(bridge_poj_dyn.time, bridge_poj_dyn.Amplitude, label='signal')
    ax.plot(bridge_poj_dyn.time, bridge_poj_dyn.Amplitude, label='filtered_signal')
    ax.plot(bridge_poj_dyn.time_cutted, bridge_poj_dyn.amplitude_envelope_cutted, label='envelope')
    LSF_fun_points = bridge_poj_dyn.fun(bridge_poj_dyn.lsq_resutls().x, bridge_poj_dyn.time_cutted, 0)
    ax.plot(bridge_poj_dyn.time_cutted, LSF_fun_points, label='linear loss')
    softL1_fun_points = bridge_poj_dyn.fun(bridge_poj_dyn.softL1_results().x, bridge_poj_dyn.time_cutted, 0)
    #softL1_fun_points = [bridge_poj_dyn.fun(bridge_poj_dyn.softL1_results().x, i, 0) for i in bridge_poj_dyn.time] # iteracja listą - 20 sek. wolniej
    ax.plot(bridge_poj_dyn.time_cutted, LSF_fun_points, label='softL1 loss')
    huber_fun_points = bridge_poj_dyn.fun(bridge_poj_dyn.huber_results().x, bridge_poj_dyn.time_cutted, 0)
    #huber_fun_points = [bridge_poj_dyn.fun(bridge_poj_dyn.huber_results().x, i, 0) for i in bridge_poj_dyn.time] # iteracja listą - 20 sek. wolniej
    ax.plot(bridge_poj_dyn.time_cutted, huber_fun_points, label='huber loss')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Rel. vert. displacement [mm]')
    plt.legend()
    plt.show()
    
    
    # Frequency domain representation - plot
    fig2, ax2 = plt.subplots()
    ax2.set_title('Fourier transform depicting the frequency components')
    ax2.plot(bridge_poj_dyn.frequecies_cutted, abs(bridge_poj_dyn.fourierTransform_cutted))
    plt.xticks([1.13])
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Amplitude')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()