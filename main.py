import pandas as pd
import default_classifier

LEARNING_RATE = 0.1
EPOCH = 5
INIT_WEIGHT = 0.2
INIT_BIAS = 0.2

# Load Datasets
df = pd.read_csv("datasets/Iris.csv")

# Split Dataset Irir-Setosa
df_setosa = df[df["Species"] == "Iris-setosa"]
setosa_train = df_setosa.iloc[0:40]
setosa_test = df_setosa.iloc[40:]

# Split Dataset Iris-versicolor
df_versicolor = df[df["Species"] == "Iris-versicolor"]
versicolor_train = df_versicolor.iloc[0:40]
versicolor_test = df_versicolor.iloc[40:]

# Combine Training Dataset
train_df = pd.concat([setosa_train, versicolor_train])
train_df["target"] = train_df.apply(
    lambda row: 0 if row["Species"] == "Iris-setosa" else 1, axis=1
)

# Combine Test Dataset
test_df = pd.concat([setosa_test, versicolor_test])
test_df["target"] = test_df.apply(
    lambda row: 0 if row["Species"] == "Iris-setosa" else 1, axis=1
)

classifier = default_classifier.DefaultClassifier()

# EPOCH 1
print("\n========================")
print("Training Epoch 1")
result_epoch_1 = classifier.fit(
    train_df,
    LEARNING_RATE,
    INIT_WEIGHT,
    INIT_BIAS,
    ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    "target",
    1,
)

print(f"Epoch 1 Weight: {result_epoch_1.weight}")
print(f"Epoch 1 Bias: {result_epoch_1.bias}")
print(f"Epoch 1 Training MSE: {result_epoch_1.MSE()}")
print(f"Epoch 1 Training Accuracy: {result_epoch_1.Accuracy()}")
print("-------------------------")

# Inference EPOCH 1
print("Inference Epoch 1")
result_inference = classifier.inference(
    test_df,
    result_epoch_1.weight,
    result_epoch_1.bias,
    ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    "target",
)
print(f"Inference Epoch 1 Accuracy: {result_inference.Accuracy()}")
print(f"Inference Epoch 1 MSE: {result_inference.MSE()}")
print("========================")
