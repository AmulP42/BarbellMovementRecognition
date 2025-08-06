import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# Loading dataframe
df = pd.read_pickle("../../data/interim/outliers_removed_chauvenets.pkl")

predictor_columns = list(df.columns[:6])
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20,5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# Data Imputation
for col in predictor_columns:
    df[col] = df[col].interpolate()
    
df.info()    

# Calculating Set Duration
duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]

duration.seconds

for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    end = df[df["set"] == s].index[-1]
    duration = end - start
    df.loc[(df["set"] == s), "duration"] = duration.seconds
    
duration_df = df.groupby(["category"])["duration"].mean()
duration_df.iloc[0] / 5
duration_df.iloc[1] / 10

# Butterworth Lowpass Filter
df_lowpass = df.copy()
LowPass = LowPassFilter()
sampling = 1000 / 200
cutoff = 1.3

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", sampling, cutoff)

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, sampling, cutoff)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]
    

# Principal Component Analysis
PCA = PrincipalComponentAnalysis()
df_pca = df_lowpass.copy()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)


# Sum of Squares
df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2 
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2 

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 15]
subset[["acc_r", "gyr_r"]].plot(subplots=True)

df_squared


# Temporal Abstraction
df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

predictor_columns = predictor_columns + ["acc_r",  "gyr_r"]

win_size = int(1000/200)

for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], win_size, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], win_size, "std")
    
df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s]
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], win_size, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], win_size, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)


# Frequency Features
df_frequency = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

sampling_rate = int(1000/200)
win_size = int(2800 / 200)

df_frequency = FreqAbs.abstract_frequency(df_frequency, ["acc_y"], win_size, sampling_rate)

df_frequency_list = []
for s in df_frequency["set"].unique():
    subset = df_frequency[df_frequency["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, win_size, sampling_rate)
    df_frequency_list.append(subset)
    
df_frequency = pd.concat(df_frequency_list).set_index("epoch (ms)", drop=True)
df_frequency.info()

df_frequency = df_frequency.dropna()

df_frequency = df_frequency.iloc[::2]


# Clustering
df_cluster = df_frequency.copy()
cluster_columns = ["acc_x", "acc_y", "acc_z"]

k_values = range(2,10)

inertia = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertia.append(kmeans.inertia_)
    
plt.figure(figsize=(10,10))
plt.plot(k_values, inertia)
plt.xlabel(k)
plt.ylabel("Sum of squared distances")
plt.show()

subset = df_cluster[cluster_columns]
kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
df_cluster["cluster"] = kmeans.fit_predict(subset)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
plt.legend()


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection="3d")
for label in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == label]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=label)
plt.legend()


# Export
df_cluster.to_pickle("../../data/interim/data_features.pkl")