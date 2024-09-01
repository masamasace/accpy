import pandas as pd
import numpy as np
from pathlib import Path
import datetime
from matplotlib import pyplot as plt
import types

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "dejavuserif" 
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = "k"

class ADXL355Data():
    def __init__(self, file_path, gain=1, trim_indice=None):
        self.file_path = Path(file_path).resolve()
        self.result_path = self.file_path.parent / "result"
        self.gain = gain
        self.trim_indice = trim_indice

        if not self.result_path.exists():
            self.result_path.mkdir()
        
        self._read_data()
        self._format_data(self.gain)

    def _read_data(self):

        self.data = pd.read_csv(self.file_path, header=None, names=['Time_ms', 'X_raw', 'Y_raw', 'Z_raw'])

    def _format_data(self, gain):

        self.data['Time_s'] = self.data['Time_ms'] / 1000
        self.data['X_gal'] = self.data['X_raw'] * gain
        self.data['Y_gal'] = self.data['Y_raw'] * gain
        self.data['Z_gal'] = self.data['Z_raw'] * gain
        
        if self.trim_indice == None:
            self.trim_indice = (0, len(self.data['Time_s']))

    def set_trim_indice(self, trim_indice):
        self.trim_indice = trim_indice
        
    def get_data(self):
        return self.data

    def export_csv(self):
        csv_file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") + "_" + self.file_path.stem + "_time_series.csv"
        file_path = self.result_path / csv_file_name

        self.data.to_csv(file_path, columns=['Time_s', 'X_gal', 'Y_gal', 'Z_gal'], index=False)

        return file_path

    def export_figure(self):

        fig_file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") + "_" + self.file_path.stem + "_time_series.svg"

        if self.trim_indice is not None:
            if len(self.trim_indice) != 2 or type(self.trim_indice) != tuple:
                raise ValueError("trim_indice must be a tuple with two elements.")
            elif type(self.trim_indice[0]) not in (int, types.NoneType) or type(self.trim_indice[1]) not in (int, types.NoneType):
                print(type(self.trim_indice[0]))
                raise ValueError("trim_indice must be a tuple with two elements of int or None.") 
            else:
                if self.trim_indice[0] == None:
                    temp_trim_indice_start = 0
                else:
                    temp_trim_indice_start = self.trim_indice[0]
                if self.trim_indice[1] == None:
                    temp_trim_indice_end = len(self.data['Time_s'])
                else:
                    temp_trim_indice_end = self.trim_indice[1]
                
                self.trim_indice = (temp_trim_indice_start, temp_trim_indice_end)

            temp_data = self.data[self.trim_indice[0]:self.trim_indice[1]].reindex()
        
        else:
            temp_data = self.data
        
        # get maximum of absolute value along each axis
        max_gal = np.max(np.abs(temp_data[['X_gal', 'Y_gal', 'Z_gal']].values))

        # get mean and std deviation value of each axis
        mean_gal = np.mean(temp_data[['X_gal', 'Y_gal', 'Z_gal']].values, axis=0)
        std_dev_gal = np.std(temp_data[['X_gal', 'Y_gal', 'Z_gal']].values, axis=0)

        fig, ax = plt.subplots(4, 1, figsize=(4, 12))

        ax[0].plot(temp_data['Time_s'], temp_data['X_gal'], label='X-axis', linewidth=0.25, color='r')
        ax[1].plot(temp_data['Time_s'], temp_data['Y_gal'], label='Y-axis', linewidth=0.25, color='g')
        ax[2].plot(temp_data['Time_s'], temp_data['Z_gal'], label='Z-axis', linewidth=0.25, color='b')

        # Fourier Transform
        N = len(temp_data['Time_s'])
        dt = temp_data['Time_s'].iloc[1] - temp_data['Time_s'].iloc[0]
        freq = np.fft.fftfreq(N, dt)
        freq = freq[:N//2]
        ax[3].plot(freq, np.abs(np.fft.fft(temp_data['X_gal']))[:N//2]/N*2, label='X-axis', linewidth=0.5, color='r', alpha=0.5)
        ax[3].plot(freq, np.abs(np.fft.fft(temp_data['Y_gal']))[:N//2]/N*2, label='Y-axis', linewidth=0.5, color='g', alpha=0.5)
        ax[3].plot(freq, np.abs(np.fft.fft(temp_data['Z_gal']))[:N//2]/N*2, label='Z-axis', linewidth=0.5, color='b', alpha=0.5)

        for i in range(4):
            if i == 0:
                ax[i].set_ylabel('X (gal)')
            if i == 1:
                ax[i].set_ylabel('Y (gal)')
            if i == 2:
                ax[i].set_ylabel('Z (gal)')
            if i != 3:
                ax[i].set_xlabel('Time (s)')
                ax[i].set_ylim(-max_gal, max_gal)
                # annonate mean and absolute value
                ax[i].annotate('Mean: {:.3f}'.format(mean_gal[i]), xy=(0.05, 0.9), xycoords='axes fraction', fontsize=8)
                ax[i].annotate('Std Dev: {:.3f}'.format(std_dev_gal[i]), xy=(0.05, 0.85), xycoords='axes fraction', fontsize=8)
                # make minor ticks available
                ax[i].minorticks_on()

            if i == 3:
                ax[i].set_ylabel('Fourier Amplitude')
                ax[i].set_xlabel('Frequency (Hz)')
                ax[i].set_yscale('log')
                ax[i].set_xscale('log')
                # set annotation

            # remove top and right spines
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)

        title_text = self.file_path.stem + " Time Series" + "\n" + "Trimmed from " + str(self.trim_indice[0]) + " to " + str(self.trim_indice[1])
        fig.suptitle(title_text, fontsize=8)

        plt.tight_layout()
        plt.savefig(self.result_path / fig_file_name, dpi=600)

        return self.result_path / fig_file_name


class ASWData():
    def __init__(self, file_path, trim_indice=None, sensor_in_use=[0, 3]):
        
        self.file_path = Path(file_path).resolve()
        self.result_path = self.file_path.parent / "result"
        self.trim_indice = trim_indice
        self.sensor_in_use = sensor_in_use
        
        self._read_data()
        self._format_data()

    def _read_data(self):
        
        encoding_chr = "shift-jis"
        self.data = pd.read_csv(self.file_path, 
                                skiprows=14,
                                names=["Time_s", "0", "1", "2", "3"],
                                encoding=encoding_chr)
        

    def _format_data(self):
        
        if self.trim_indice == None:
            self.trim_indice = (0, len(self.data['Time_s']))
    
    def set_trim_indice(self, trim_indice):
        self.trim_indice = trim_indice
    
    def get_data(self):
        return self.data
    
    def export_csv(self):
        csv_file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") + "_" + self.file_path.stem + "_time_series.csv"
        file_path = self.result_path / csv_file_name

        temp_export_columns = ['Time_s'] + [str(i) for i in self.sensor_in_use]
        self.data.to_csv(file_path, columns=temp_export_columns, index=False)

        return file_path

    def export_figure(self):

        fig_file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") + "_" + self.file_path.stem + "_time_series.svg"

        if self.trim_indice is not None:
            if len(self.trim_indice) != 2 or type(self.trim_indice) != tuple:
                raise ValueError("trim_indice must be a tuple with two elements.")
            elif type(self.trim_indice[0]) not in (int, types.NoneType) or type(self.trim_indice[1]) not in (int, types.NoneType):
                print(type(self.trim_indice[0]))
                raise ValueError("trim_indice must be a tuple with two elements of int or None.") 
            else:
                if self.trim_indice[0] == None:
                    temp_trim_indice_start = 0
                else:
                    temp_trim_indice_start = self.trim_indice[0]
                if self.trim_indice[1] == None:
                    temp_trim_indice_end = len(self.data['Time_s'])
                else:
                    temp_trim_indice_end = self.trim_indice[1]
                
                self.trim_indice = (temp_trim_indice_start, temp_trim_indice_end)

            temp_data = self.data[self.trim_indice[0]:self.trim_indice[1]].reindex()
        
        else:
            temp_data = self.data
        
        temp_column_in_use = ["Time_s"] + [str(i) for i in self.sensor_in_use]
        temp_data = temp_data[temp_column_in_use]
        
        # get maximum of absolute value along each axis
        max_gal = np.max(np.abs(temp_data[temp_column_in_use].values))
        
        # get mean and std deviation value of each axis
        mean_gal = np.mean(temp_data[temp_column_in_use].values, axis=0)
        std_dev_gal = np.std(temp_data[temp_column_in_use].values, axis=0)
        
        fig, ax = plt.subplots(len(temp_column_in_use), 1, figsize=(4, 3 * (len(temp_column_in_use))))

        for i in range(len(temp_column_in_use)):
            
            if i != len(temp_column_in_use) - 1:
                
                ax[i].plot(temp_data['Time_s'], temp_data[temp_column_in_use[i+1]], label=temp_column_in_use[i+1], linewidth=0.25)
                ax[i].set_ylim(-max_gal, max_gal)
                ax[i].set_xlabel('Time (s)')
                ax[i].set_ylabel("Sensor " + temp_column_in_use[i+1] + " (gal)")
                ax[i].legend()
                ax[i].annotate('Mean: {:.3f}'.format(mean_gal[i+1]), xy=(0.05, 0.9), xycoords='axes fraction', fontsize=8)
                ax[i].annotate('Std Dev: {:.3f}'.format(std_dev_gal[i+1]), xy=(0.05, 0.85), xycoords='axes fraction', fontsize=8)
                
            else:
                ax[i].set_ylabel('Fourier Amplitude')
                ax[i].set_xlabel('Frequency (Hz)')
                
                N = len(temp_data['Time_s'])
                dt = temp_data['Time_s'].iloc[1] - temp_data['Time_s'].iloc[0]
                freq = np.fft.fftfreq(N, dt)
                freq = freq[:N//2]
                
                for column in temp_column_in_use[1:]:
                    
                    ax[i].plot(freq, np.abs(np.fft.fft(temp_data[column]))[:N//2]/N*2, label=column, linewidth=0.5, alpha=0.5)
                    
                ax[i].set_yscale('log')
                ax[i].set_xscale('log')

            # remove top and right spines
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            
        title_text = self.file_path.stem + " Time Series" + "\n" + "Trimmed from " + str(self.trim_indice[0]) + " to " + str(self.trim_indice[1])
        fig.suptitle(title_text, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.result_path / fig_file_name, dpi=600)
        
        return self.result_path / fig_file_name
        


