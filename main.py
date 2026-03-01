import pandas as pd
import default_classifier

LEARNING_RATE = 0.01
EPOCH = 5
INIT_WEIGHT = 0.2
INIT_BIAS = 0.2

df = pd.read_csv("datasets/Iris.csv")
print(df.head())

df_setosa = df[df["Species"] == "Iris-setosa"]
setosa_train = df_setosa.iloc[0:40]
setosa_test = df_setosa.iloc[40:]

df_versicolor = df[df["Species"] == "Iris-versicolor"]
versicolor_train = df_versicolor.iloc[0:40]
versicolor_test = df_versicolor.iloc[40:]

train_df = pd.concat([setosa_train, versicolor_train])

classifier = default_classifier.DefaultClassifier()
classifier.fit(
    train_df,
    INIT_WEIGHT,
    INIT_BIAS,
    ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
)
