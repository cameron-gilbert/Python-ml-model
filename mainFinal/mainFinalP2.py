import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score

# Load your data from CSV files
tobyJumpFile = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\Toby Jumping Data\tobyJumping.csv"
tobyWalkFile = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\Toby Walking Data\tobyWalking.csv"
tobyJumpData = pd.read_csv(tobyJumpFile)
tobyWalkData = pd.read_csv(tobyWalkFile)

for index, value in tobyJumpData.iloc[17886:23845, 0].items():
    # Do something with each value
    tobyJumpData.at[index, tobyJumpData.columns[0]] = value + 180

for index, value in tobyJumpData.iloc[23845:29805, 0].items():
    # Do something with each value
    tobyJumpData.at[index, tobyJumpData.columns[0]] = value + 240

tobyJumpData['Label'] = 1
tobyWalkData['Label'] = 0

tobyData = pd.concat([tobyJumpData, tobyWalkData])

def correctTime(data, number):
    # Iterate over each row in the first column
    for index, value in data.iloc[:, 0].items():
        # Do something with each value
        data.at[index, data.columns[0]] = value + 60 * number


camJumpFileHold = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\Cam Jump holding\camJumpingHolding.csv"
camJumpFileLF = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\Cam Jumping LF\camJumpingLF.csv"
camJumpFileLR = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\Cam Jumping LR\camJumpingLR.csv"
camJumpFileRF = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\Cam Jumping RF\camJumpingRF.csv"
camJumpFileRR = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\Cam Jumping RR\camJumpingRR.csv"
camWalkFileHold = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\Cam Walking holding\camWalkingHolding.csv"
camWalkFileLF = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\Cam Walking LF\camWalkingLF.csv"
camWalkFileLR = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\Cam Walking LR\camWalkingLR.csv"
camWalkFileRF = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\Cam Walking RF\camWalkingRF.csv"
camWalkFileRR = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\Cam walking RR\camWalkingRR.csv"
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

correctTime(camWalkDataRR, 4)
correctTime(camWalkDataRF, 3)
correctTime(camWalkDataLR, 2)
correctTime(camWalkDataLF, 1)

correctTime(camJumpDataRR, 4)
correctTime(camJumpDataRF, 3)
correctTime(camJumpDataLR, 2)
correctTime(camJumpDataLF, 1)

camJumpData = pd.concat([camJumpDataHold, camJumpDataLF, camJumpDataLR, camJumpDataRF, camJumpDataRR])
camWalkData = pd.concat([camWalkDataHold, camWalkDataLF, camWalkDataLR, camWalkDataRF, camWalkDataRR])

camJumpData['Label'] = 1
camWalkData['Label'] = 0

camData = pd.concat([camJumpData, camWalkData])

ayoJumpFileHold = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\ayoJumpingHolding.csv"
ayoJumpFileLF = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\ayoJumpingLF.csv"
ayoJumpFileLR = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\ayoJumpingLR.csv"
ayoJumpFileRF = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\ayoJumpingRF.csv"
ayoJumpFileRR = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\ayoJumpingRR.csv"
ayoWalkFileHold = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\ayoWalkingHolding.csv"
ayoWalkFileLF = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\ayoWalkingLF.csv"
ayoWalkFileLR = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\ayoWalkingLR.csv"
ayoWalkFileRF = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\ayoWalkingLR.csv"
ayoWalkFileRR = r"C:\Users\Cameron\PycharmProjects\mainFinal\ELEC292FINAL\ayoWalkingLR.csv"
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

correctTime(ayoWalkDataRR, 4)
correctTime(ayoWalkDataRF, 3)
correctTime(ayoWalkDataLR, 2)
correctTime(ayoWalkDataLF, 1)

correctTime(ayoJumpDataRR, 4)
correctTime(ayoJumpDataRF, 3)
correctTime(ayoJumpDataLR, 2)
correctTime(ayoJumpDataLF, 1)

ayoJumpData = pd.concat([ayoJumpDataHold, ayoJumpDataLF, ayoJumpDataLR, ayoJumpDataRF, ayoJumpDataRR])
ayoWalkData = pd.concat([ayoWalkDataHold, ayoWalkDataLF, ayoWalkDataLR, ayoWalkDataRF, ayoWalkDataRR])

ayoJumpData['Label'] = 1
ayoWalkData['Label'] = 0

ayoData = pd.concat([ayoJumpData, ayoWalkData])
fullData = pd.concat([ayoData, tobyData, camData])

# Segment the data into 5-second windows
window_size = 5 * 50  # Assuming 50 Hz sampling rate, 5 seconds = 250 samples

# Function to segment the data

def segment_data(data):
    segmented_data = []
    num_windows = len(data) // window_size
    for i in range(num_windows):
        # Slice the entire DataFrame by rows for each window
        window = data.iloc[i * window_size:(i + 1) * window_size, :]
        segmented_data.append(window)
    return segmented_data


# Segment data from jumping and walking datasets
segmentedData = segment_data(fullData)

# Shuffle the segmented data
segmentedShuffledData = shuffle(segmentedData)

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

#VISUALIZATION
def scatterPlot (data, activity, person):

    plt.figure(figsize=(10,6))
    time = data.iloc[:, 0]
    xAcceleration = data.iloc[:, 1]
    yAcceleration = data.iloc[:, 2]
    zAcceleration = data.iloc[:, 3]

    plt.plot(time, xAcceleration, label='X Acceleration')
    plt.plot(time, yAcceleration, label='Y Acceleration')
    plt.plot(time, zAcceleration, label='Z Acceleration')
    plt.title(f'{person} - {activity} Acceleration versus Time')

    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.legend()
    plt.show()


def scatterPlotAcc (data, activity, person):

    plt.figure(figsize=(10,6))
    time = data.iloc[:, 0]
    acceleration = data.iloc[:, 4]

    plt.plot(time, acceleration, label='Acceleration')
    plt.title(f'{person} - {activity} Acceleration versus Time')

    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.legend()
    plt.show()


def walkVjumpPlot (jumpData, walkData, person):
    plt.figure(figsize=(10,6))
    time1 = jumpData.iloc[:, 0]
    time2 = walkData.iloc[:, 0]

    jumpAcceleration = jumpData.iloc[:, 4]
    walkAcceleration = walkData.iloc[:, 4]

    plt.plot(time1, jumpAcceleration, label='Jumping Acceleration')
    plt.plot(time2, walkAcceleration, label='Walking Acceleration')

    plt.title(f'{person} Jump Acceleration versus Walk Acceleration over Time')

    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.legend()
    plt.show()


'''
PRE-PROCESSING
APPLY MOVING AVERAGE
using window size of 60 based of visual observation and min_periods avoids nan values
columns where data is observed(ie x,y,z)
'''
column_indices = [1, 2, 3] #indices of x,y,z


def moving_average(data):
    data_copy = data.copy() #copy data so that orginal files dont get affected
    data_copy.iloc[:, column_indices] = data_copy.iloc[:, column_indices].rolling(window=500, min_periods=1).mean()
    '''Apply rolling average on each column by itself. Uses windowsize of 60 and min_periiods=1 so that if 
    there isn't enough data the average will still be applied with what it has'''
    return data_copy


'''Here we take the shuffled data and apply the moving average to each window'''
segmentedShuffledData = [moving_average(segment) for segment in segmentedShuffledData]

'''This next function extracts the features of each x y and z from a segment'''


def extract_features_from_segment(segment):
    #Extract features for X acceleration
    #X acceleration data is at index 1
    x = segment.iloc[:, 1]
    features_x = {
        'mean_x': x.mean(),             #mean
        'std_x': x.std(),      #standard deviation
        'max_x': x.max(),       #max
        'min_x': x.min(),        #min
        'range_x': x.max() - x.min(),       #range
        'median_x': x.median(),     #median
        'variance_x': x.var(),    #variance
        'skew_x': x.skew(),  #skew from class
        'kurtosis_x': x.kurt(),        # kurtosis from class
        'iqr_x': np.subtract(*np.percentile(x, [75, 25])),   #interquartile range
    }

    '''Extract features for Y acceleration'''
    y = segment.iloc[:, 2]  # Y acceleration data is at index 2
    features_y = {
        'mean_y': y.mean(),
        'std_y': y.std(),
        'max_y': y.max(),
        'min_y': y.min(),
        'range_y': y.max() - y.min(),
        'median_y': y.median(),
        'variance_y': y.var(),
        'skew_y': y.skew(),
        'kurtosis_y': y.kurt(),
        'iqr_y': np.subtract(*np.percentile(y, [75, 25])),
    }

    # Extract features for Z acceleration
    z = segment.iloc[:, 3]  # Z acceleration data is at index 3
    features_z = {
        'mean_z': z.mean(),
        'std_z': z.std(),
        'max_z': z.max(),
        'min_z': z.min(),
        'range_z': z.max() - z.min(),
        'median_z': z.median(),
        'variance_z': z.var(),
        'skew_z': z.skew(),
        'kurtosis_z': z.kurt(),
        'iqr_z': np.subtract(*np.percentile(z, [75, 25])),
    }
    '''since the label of the segment should be the same in every row, we can just take the value from the first row which 
    will represent the whole segment'''
    label = segment.iloc[0, 5]
    '''Combine all features into a single dictionary adding on the label to the last column'''
    features = {**features_x, **features_y, **features_z, 'label': label}

    return features


'''following function processes a list of segmented data '''


def preprocess_data(segmented_data):
    # Create a DataFrame to hold all features for each segment
    all_features = [extract_features_from_segment(segment) for segment in segmented_data]

    '''Convert the list of features dictionaries to a DataFrame'''
    features_df = pd.DataFrame(all_features)

    '''Separate the features and labels'''
    labels = features_df['label']
    features = features_df.drop('label', axis=1)

    '''Normalize the features using z score standardization. First get the import '''
    scaler = preprocessing.StandardScaler()
    '''next, fit the scalar to the data, while also performing its transformation'''
    normalized_features = scaler.fit_transform(features)
    normalized_features_df = pd.DataFrame(normalized_features, columns=features.columns)

    '''Reattach the labels'''
    normalized_features_df['label'] = labels
    
    '''Drop rows with NaN values as they will not work when classifying '''
    normalized_features_df = normalized_features_df.dropna()
    '''split into data and labels before splitting for regression'''
    data = normalized_features_df.drop('label', axis=1)
    labels = normalized_features_df['label'].values.reshape(-1, 1)
    return data, labels


'''call preprocess function on the shuffled segmented data'''
X, y = preprocess_data(segmentedShuffledData)
'''split the resulting data into 90%training and 10%testing'''
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.10, shuffle=True)

#Step 6 TRAINING CLASSIFER
'''uses same steps from class slides'''
'''using logistic regression import, we use 1000 iterations for training the classifier'''
l_reg = LogisticRegression(max_iter=1000)

clf = make_pipeline(StandardScaler(), l_reg)

'''Training'''
'''.ravel must be used as .fit requires a 1D array, but Y_train is not 1D so it must flatten out'''
clf.fit(X_train, Y_train.ravel())

'''following block of code was used to check working condition of the classifier'''
# Make predictions
y_pred = clf.predict(X_test)
y_clf_prob = clf.predict_proba(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f'Accuracy of the model: {accuracy}')
recall = recall_score(Y_test, y_pred)
print('recall is ', recall)

'''Next plot Training Curves using import function'''
train_sizes, train_scores, test_scores = learning_curve(
    clf, X, y.ravel(), cv=5, n_jobs=-1,
    train_sizes=np.linspace(.1, 1.0, 10),
    scoring='accuracy'
)
''''''
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.title('Learning Curve for Logistic Regression')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend(loc="best")
plt.grid(True)
plt.show()

'''the following function is used in mapping predictions in each segment to the original csv input data'''


def map_predictions_to_original_data(input_data, predictions, window_size):
    # Initialize the prediction column in the original data
    input_data['Predicted Label'] = np.nan

    '''Iterate over each prediction and map it to the corresponding rows in the original data
    this was done knowing the way that we created segments in the segment_data function. Meaning here 
    we are just operating in the opposite fashion.'''
    for i, prediction in enumerate(predictions):
        start_index = i * window_size
        end_index = start_index + window_size
        input_data.loc[start_index:end_index, 'Predicted Label'] = prediction

    return input_data


# Step 7 CREATION OF GUI
def classify_and_output_csv():
    global clf
    print("classify output csv is called")

    '''user inputs CSV file'''
    input_file_path = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=[("CSV files", "*.csv")]
    )
    if not input_file_path:
        return  # If no file is selected, return early

    try:
        # Load the input data
        org_data = pd.read_csv(input_file_path)
        '''ensure org data doesnt get affected'''
        input_data = org_data.copy()
        '''Here we add a dummy placeholder for a label so that we can use functions from above with ease.
         In the way that the past calculations were performed above, the label never had any affect so this works'''
        input_data['Label'] = 0
        '''segment the data'''
        seg_input_data = segment_data(input_data)
        '''apply moving average in same way'''
        input_data_avg = [moving_average(segment) for segment in seg_input_data]
        '''preprocess in same way as when making the model to ensure functionality
        It is also in this step that we are able to take out the dummy labels naturally by the way that 
        the preprocessing splits the information'''
        data, labels = preprocess_data(input_data_avg)

        '''Predict the classes using the already trained model'''
        predictions = clf.predict(data)

        '''Map the predictions to the original data using the function described above'''
        mapped_data = map_predictions_to_original_data(org_data, predictions, window_size)

        '''Ask the user to select the output file path'''
        output_file_path = filedialog.asksaveasfilename(
            title="Save the CSV file with predictions",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )

        if output_file_path:
            # Save the dataframe with predictions to a new CSV file
            mapped_data.to_csv(output_file_path, index=False)
            messagebox.showinfo("Success", "File with predictions saved successfully!")

        # Plotting
        plt.figure()
        plt.plot(range(len(predictions)), predictions, label='Predicted Activity')
        plt.xlabel('Window Index')
        plt.ylabel('Activity (0-Walking, 1-Jumping)')
        plt.title('Predicted Walking (0) or Jumping (1) Over Time')
        plt.legend()
        plt.show()


    except Exception as e:
        messagebox.showerror("Error", f"Failed to classify and save data.\nError: {e}")


# GUI setup
app = tk.Tk()
app.title("Activity Classifier Application")

# Changes
app.geometry("600x400")  # Set a default size for the app

# Define a style for the app
style = ttk.Style(app)
style.configure('TButton', font=('Helvetica', 12), padding=10)
style.configure('TFrame', background='light grey')

# Create frames to organize the layout
header_frame = ttk.Frame(app, padding="10 10 10 10", style='TFrame')
header_frame.pack(side=tk.TOP, fill=tk.X, expand=False)

content_frame = ttk.Frame(app, padding="10 10 10 10", style='TFrame')
content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Header label
header_label = ttk.Label(header_frame, text="Activity Classification", font=('Helvetica', 16, 'bold'),
                         background='light grey')
header_label.pack(side=tk.TOP, pady=10)

# Input frame within content frame for better organization
input_frame = ttk.Frame(content_frame, padding="10 10 10 10", style='TFrame')
input_frame.pack(side=tk.TOP, fill=tk.X, expand=False)

# Input label and entry field
input_label = ttk.Label(input_frame, text="Input CSV File:", background='light grey')
input_label.pack(side=tk.LEFT, padx=5, pady=5)

input_entry = ttk.Entry(input_frame, width=50)
input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

# Button frame within content frame for better organization
button_frame = ttk.Frame(content_frame, padding="10 10 10 10", style='TFrame')
button_frame.pack(side=tk.TOP, fill=tk.X, expand=False)

# Changes

# Create a button that calls the classify_and_output_csv function
classify_button = tk.Button(app, text="Classify Data", command=classify_and_output_csv)
classify_button.pack(pady=20)

# Changes
# Status bar at the bottom
status_bar = ttk.Label(app, text="Ready", relief=tk.SUNKEN, anchor=tk.W, padding=2, background='light grey')
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Ensure that the content frame expands with the window
content_frame.rowconfigure(1, weight=1)
content_frame.columnconfigure(0, weight=1)

# Changes

# Run the GUI event loop
app.mainloop()