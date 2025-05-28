"""
According to: Gary Weiss, WISDM Smartphone and Smartwatch Activity and Biometrics Dataset, 2019.
"""
import torch
import os
import pandas as pd
import numpy as np


class WISDM:
    """Class to design a WISDM Dataset."""

    def __init__(self, data_path):
        """Initialisation of the class (constructor). It prepares the data to be used for training and validation."""
        # Input:
        # mode; string, train or test data

        # Load the data
        self.column_names = ["ID", "Activity", "TimeStamp", "X", "Y", "Z"]
        self.acitvity_names = ["Walking", "Jogging", "Stairs", "Sitting", "Standing",  "Typing", "Brush teeth", "Eat Soup", "Eat chips", "Eat Pasta", "Drinking", "Eat Sandwich", "Kicking", "Catch", "Dribblilg", "Writing", "Clapping", "Fold Clothes"]

        self.activity_dic = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "O": 13, "P": 14, "Q": 15, "R": 16, "S": 17}

        # Inverted dictionary for reconversion
        self.activity_dic_inv = {item: element for element, item in self.activity_dic.items()}

        self.folder = data_path + "./wisdm-dataset/wisdm-dataset/raw/watch/gyro/"
        self.filelist = [txt for txt in os.listdir(self.folder) if txt[-4:] == ".txt"]
        self.data_tensor = []
        self.data_tensor_raw = []

        self.__create_tensor()

        self.predfname = None
        
    def __create_tensor(self):
        """This method combines all text files into one big tensor."""

        for txt in self.filelist:
            self.data = pd.read_csv(self.folder + txt, header = None, names = self.column_names, comment = ";") # load data

            # Replaced the activity description currently with letters with numbers
            self.data["Activity"] = self.data["Activity"].map(self.activity_dic)

            # safe the raw data in a tensor
            if (self.data_tensor_raw == []):
                self.data_tensor_raw = torch.tensor(self.data.values).float()
            else:
                self.data_tensor_raw = torch.cat((self.data_tensor_raw, torch.tensor(self.data.values).float()))

            self.__normalize_feature()  # normalizes all features

            # safe the normalized data in a tensor
            if (self.data_tensor == []):
                self.data_tensor = torch.tensor(self.data.values).float()
            else:
                self.data_tensor = torch.cat((self.data_tensor, torch.tensor(self.data.values).float()))

    def __normalize_feature(self):
        """This method normalizes all features."""

        for dim in ["X", "Y", "Z"]:
            # normalize the data
            mue = np.mean(self.data[dim])     # Mean
            sigma = np.std(self.data[dim])    # Standard deviation
            self.data[dim] = (self.data[dim] - mue) / sigma

    def dataloading(self, window_size, overlap):
        """This method fills the DataLoader."""
        data_tensor = self.slid_win(self.data_tensor, window_size, overlap)

        return data_tensor

    def visualisation(self):
        """This method visualises the data."""

        self.__vis_data_points_per_category()   # Number of data points in each category as bar chart
        self.__vis_sample_series_per_category() # Sample data series for all six categories
        
    def __vis_data_points_per_category(self):
        """This method displays the number of data points in each category as a bar chart."""

        activity_counts = torch.unique(self.data_tensor[:, 1].long(), sorted = True, return_counts = True)
        mean = torch.mean(activity_counts[1].float())
        
        import matplotlib.pyplot as plt
        plt.rcParams["figure.figsize"] = (12, 7)
        plt.bar(self.acitvity_names, activity_counts[1], label = "Number of data points")
        plt.axhline(mean, label = "Mean", color = "red")
        plt.title("Number of datapoints by Activities")
        plt.legend()
        plt.savefig("Number_of_datapoints_by_Activities.png")
        plt.show()

    def __vis_sample_series_per_category(self):
        """This method visualises sample data series for all six categories."""
        
        length = 200
        labels = ["x-signal", "y-signal", "z-signal"]
        x_values = np.linspace(0.0, length * 0.05, length)

        fig, axes = plt.subplots(3, 2, sharex = True, figsize = (18, 9))

        for i in range(6):  # Fill all subplots
            start = i * 2400
            tensorxyz = self.data_tensor_raw[start:start + length, 2:5]   # data for this plot

            if (i < 3):
                row = i
                col = 0
            else:
                row = i - 3
                col = 1
                
            axes[row, col].plot(x_values, tensorxyz)
            axes[row, col].set_title(self.acitvity_names[i])
            axes[row, col].grid()

        fig.legend(labels)
        import matplotlib.pyplot as plt
        plt.setp(axes[-1, :], xlabel = "Time [s]")
        plt.suptitle("Sample data series of each category")
        plt.savefig("Sample_data_series_of_each_category.png")
        plt.show()
    
    def slid_win(self, data, window_size, step_size):
        """This method implements a sliding window."""
        output = data.unfold(0, window_size, step_size).transpose(1, 2)
        mask = torch.ones(output.shape[0], dtype=torch.bool)
        for i in range(output.shape[0]):  # remove the samples that contains clips of two activities
            if output[i, 0, 1] != output[i, -1, 1]:
                mask[i] = False
        output = output[mask]
        return output


