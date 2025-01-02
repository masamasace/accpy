import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import os 

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "dejavuserif" 
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = "k"

class AccelerometerData():

    def __init__(self):
        
        self.data_raw = None
        self.data = None
        self.data_path = None
        self.gain = None
        # 基線補正を計算するために最初から何個のデータを使うか
        self.axis_reverse = None
        self.bc_index = None
        self.value_offset = None
        self.time_offset = None
        self.time_range = None
        self.stats = None
        self.fft = None
        self.result_path = None
    

    def get_data(self):
        return self.data
    
    def get_stats(self):
        return self.stats
    
    def get_fft(self):
        return self.fft

    def clear_data(self):
        self.data_raw = None
        self.data = None
        self.stats = None
        self.fft = None
    

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError("No such attribute: {}".format(key))
        
        self._check_value_type()

        if self.data is not None:

            self.original_data = self.data.copy()
            self.data = self._modify_data(self.data)
            self._calc_stats_fft(self.data)
        
        elif self.data_path is not None:
            
            self.data_path = self._check_path_escape_char(self.data_path)
            
            self.data_raw = self._read_data(self.data_path)
            self.data = self._modify_data(self.data_raw)
            self._calc_stats_fft(self.data)
    

    def _check_path_escape_char(self, path):

        path = str(path)

        if Path(path).resolve().exists():
            return Path(path).resolve()
        
        else:
            if os.name == "posix":
                path = path.replace("\\", "/")
            elif os.name == "nt":
                path = path.replace("/", "\\")
            else:
                raise OSError("Unknown OS: {}".format(os.name))

            if Path(path).resolve().exists():
                return Path(path).resolve()
            else:
                raise FileNotFoundError("No such file or directory: {}".format(path))
        

    def _read_data(self, data_path):
        
        msg = "This method must be implemented in the subclass."
        raise NotImplementedError(msg)
    
    def _modify_data(self, data):

        data = data.copy()

        if self.time_range is not None:
            data = data[(data["time"] >= self.time_range[0]) & (data["time"] <= self.time_range[1])]
            data["time"] = data["time"] - data["time"].iloc[0]
    
        if self.time_offset is not None:
            data["time"] = data["time"] - self.time_offset
    
        data = data.reset_index(drop=True)

        if self.axis_reverse is not None:
            data.iloc[:, 1:] = data.iloc[:, 1:] * (np.array(self.axis_reverse) * 2 - 1)


        if self.gain is not None:
            data.iloc[:, 1:] = data.iloc[:, 1:] * self.gain

        if self.value_offset is not None:
            data.iloc[:, 1:] = data.iloc[:, 1:] - self.value_offset

        if self.bc_index is not None:
            data.iloc[:, 1:] = data.iloc[:, 1:] - data.iloc[:self.bc_index, 1:].mean()

        return data


    def _calc_stats_fft(self, data):

        self.stats = self._calc_stats(data)
        self.fft = self._calc_fft(data)


    def _calc_stats(self, data):

        stats = {
            "max": data.loc[data.iloc[:, 1:].abs().idxmax(axis=0)].reset_index(drop=True),
            "mean": data[1:].mean(),
            "std": data[1:].std()
        }

        return stats


    def _calc_fft(self, data):

        print(data)

        N = len(data)
        dt = data["time"].iloc[1] - data["time"].iloc[0]
        freq = np.fft.rfftfreq(N, dt)

        fft = pd.DataFrame(freq, columns=["freq"])
        for column in data.columns[1:]:
            fft[column] = np.abs(np.fft.rfft(data[column])) / N * 2 * data["time"].iloc[-1]
        
        return fft


    def _check_value_type(self):

        self._check_value_type_base(self.data_raw, pd.DataFrame)
        self._check_value_type_base(self.data, pd.DataFrame)
        self._check_value_type_base(self.data_path, (str, Path))
        self._check_value_type_base(self.gain, (float, int))
        self._check_value_type_base(self.axis_reverse, tuple)
        self._check_value_type_base(self.bc_index, int)
        self._check_value_type_base(self.value_offset, (float, int))
        self._check_value_type_base(self.time_offset, (float, int))
        self._check_value_type_base(self.time_range, tuple)
        self._check_value_type_base(self.stats, dict)
        self._check_value_type_base(self.fft, pd.DataFrame)
        self._check_value_type_base(self.result_path, (str, Path))


    def _check_value_type_base(self, value, value_types):
        if value is not None and not isinstance(value, value_types):
            raise TypeError("Invalid value type of {}: {}".format(value, type(value)))

class ADXL355Data(AccelerometerData):
    def _init__(self):
        
        super().__init__()

    def _read_data(self, data_path):

        data = pd.read_csv(data_path,
                                header=None,
                                names=['time', 0, 1, 2],
                                dtype=float)
        
        data['time'] = data['time'] / 1000

        return data       


# TODO: separate adxl355 and asw data class
class ASWData(AccelerometerData):
    def __init__(self):
        
        super().__init__()
    
    def _read_data(self, data_path):

        data = pd.read_csv(data_path,
                                skiprows=14,
                                names=["time", 0, 1, 2, 3],
                                encoding="shift-jis")
        
        return data