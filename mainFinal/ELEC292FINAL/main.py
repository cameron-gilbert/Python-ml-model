import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Load your data from CSV files
tobyJumpFile = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\Toby Jumping Data\tobyJumping.csv"
tobyWalkFile = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\Toby Walking Data\tobyWalking.csv"
tobyJumpData = pd.read_csv(tobyJumpFile)
tobyWalkData = pd.read_csv(tobyWalkFile)

tobyJumpData['Label'] = 1
tobyWalkData['Label'] = 0

tobyData = pd.concat([tobyJumpData, tobyWalkData])

camJumpFileHold = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\Cam Jump holding\camJumpingHolding.csv"
camJumpFileLF = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\Cam Jumping LF\camJumpingLF.csv"
camJumpFileLR = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\Cam Jumping LR\camJumpingLR.csv"
camJumpFileRF = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\Cam Jumping RF\camJumpingRF.csv"
camJumpFileRR = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\Cam Jumping RR\camJumpingRR.csv"
camWalkFileHold = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\Cam Walking holding\camWalkingHolding.csv"
camWalkFileLF = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\Cam Walking LF\camWalkingLF.csv"
camWalkFileLR = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\Cam Walking LR\camWalkingLR.csv"
camWalkFileRF = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\Cam Walking RF\camWalkingRF.csv"
camWalkFileRR = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\Cam walking RR\camWalkingRR.csv"
camJumpDataHold = pd.read_csv(camJumpFileHold)
camJumpDataLF = pd.read_csv(camJumpFileLF)
camJumpDataLR = pd.read_csv(camJumpFileLR)
camJumpDataRF = pd.read_csv(camJumpFileRF)
camJumpDataRR = pd.read_csv(camJumpFileRR)
camWalkDataHold = pd.read_csv(camWalkFileHold)
camWalkDataLF = pd.read_csv(camWalkFileLF)
camWalkDataLR = pd.read_csv(camWalkFileLR)
camWalkDataRF = pd.read_csv(camWalkFileRF)
camWalkDataRR = pd.read_csv(camWalkFileRR)
camJumpData = pd.concat([camJumpDataHold, camJumpDataLF, camJumpDataLR, camJumpDataRF, camJumpDataRR])
camWalkData = pd.concat([camWalkDataHold, camWalkDataLF, camWalkDataLR, camWalkDataRF, camJumpDataRR])

camJumpData['Label'] = 1
camWalkData['Label'] = 0

camData = pd.concat([camJumpData, camWalkData])

ayoJumpFileHold = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\ayoJumpingHolding.csv"
ayoJumpFileLF = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\ayoJumpingLF.csv"
ayoJumpFileLR = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\ayoJumpingLR.csv"
ayoJumpFileRF = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\ayoJumpingRF.csv"
ayoJumpFileRR = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\ayoJumpingRR.csv"
ayoWalkFileHold = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\ayoWalkingHolding.csv"
ayoWalkFileLF = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\ayoWalkingLF.csv"
ayoWalkFileLR = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\ayoWalkingLR.csv"
ayoWalkFileRF = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\ayoWalkingLR.csv"
ayoWalkFileRR = r"C:\Users\Toby Salamon\PycharmProjects\ELEC292FINAL\ayoWalkingLR.csv"
ayoJumpDataHold = pd.read_csv(ayoJumpFileHold)
ayoJumpDataLF = pd.read_csv(ayoJumpFileLF)
ayoJumpDataLR = pd.read_csv(ayoJumpFileLR)
ayoJumpDataRF = pd.read_csv(ayoJumpFileRF)
ayoJumpDataRR = pd.read_csv(ayoJumpFileRR)
ayoWalkDataHold = pd.read_csv(ayoWalkFileHold)
ayoWalkDataLF = pd.read_csv(ayoWalkFileLF)
ayoWalkDataLR = pd.read_csv(ayoWalkFileLR)
ayoWalkDataRF = pd.read_csv(ayoWalkFileRF)
ayoWalkDataRR = pd.read_csv(ayoWalkFileRR)
ayoJumpData = pd.concat([ayoJumpDataHold, ayoJumpDataLF, ayoJumpDataLR, ayoJumpDataRF, ayoJumpDataRR])
ayoWalkData = pd.concat([ayoWalkDataHold, ayoWalkDataLF, ayoWalkDataLR, ayoWalkDataRF, ayoJumpDataRR])

ayoJumpData['Label'] = 1
ayoWalkData['Label'] = 0

ayoData = pd.concat([ayoJumpData, ayoWalkData])

fullData = pd.concat([ayoData, tobyData, camData])

# Segment the data into 5-second windows
window_size = 5 * 50  # Assuming 50 Hz sampling rate, 5 seconds = 250 samples

# Function to segment the data
def segment_data(data):
    segmented_data = []
    for column in data.columns:
        signal = data[column]
        num_windows = len(signal) // window_size
        for i in range(num_windows):
            window = signal[i * window_size: (i + 1) * window_size]
            segmented_data.append(window)
    return segmented_data


# Segment data from jumping and walking datasets
segmentedData = segment_data(fullData)

# Shuffle the segmented data
segmentedShuffledData= shuffle(segmentedData)

# Split into training and test sets
train_data, test_data = train_test_split(segmentedShuffledData, test_size=0.1, random_state=42)

# Store the data in HDF5 file
filename_hdf5 = 'segmented_data.h5'
with h5py.File(filename_hdf5, 'w') as hf:
    #create branches for storing participants' data
    group_toby = hf.create_group('Toby Data')
    group_toby.create_dataset('Toby Jumping', data=tobyJumpData)
    group_toby.create_dataset('Toby Walking', data=tobyWalkData)
    group_cam = hf.create_group('Cam Data')
    group_cam.create_dataset('Cam Jumping', data=camJumpData)
    group_cam.create_dataset('Cam Walking', data=camWalkData)
    group_ayo = hf.create_group('Ayo Data')
    group_ayo.create_dataset('Ayo Jumping', data=ayoJumpData)
    group_ayo.create_dataset('Ayo Walking', data=ayoWalkData)

    # Create groups for jumping and walking
    group_train_test = hf.create_group('Dataset')

    group_train_test.create_dataset('train_data', data=train_data)
    group_train_test.create_dataset('test_data', data=test_data)


print("Segmented, shuffled, and split data stored in HDF5 file successfully.")