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

print(f"Epoch 1 Weight: {result_epoch_1.GetWeight()}")
print(f"Epoch 1 Bias: {result_epoch_1.GetBias()}")
print(f"Epoch 1 Training MSE: {result_epoch_1.GetMSEPerEpoch()}")
print(f"Epoch 1 Training Accuracy: {result_epoch_1.GetAccuracyPerEpoch()}")
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

# EPOCH 2
print("\n========================")
print("Training Epoch 2")
result_epoch_2 = classifier.fit(
    train_df,
    LEARNING_RATE,
    INIT_WEIGHT,
    INIT_BIAS,
    ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    "target",
    2,
)

print(f"Epoch 2 Weight: {result_epoch_2.GetWeight()}")
print(f"Epoch 2 Bias: {result_epoch_2.GetBias()}")
print(f"Epoch 2 Training MSE: {result_epoch_2.GetMSEPerEpoch()}")
print(f"Epoch 2 Training Accuracy: {result_epoch_2.GetAccuracyPerEpoch()}")
print("-------------------------")

# Inference EPOCH 2
print("Inference Epoch 2")
result_inference = classifier.inference(
    test_df,
    result_epoch_2.weight,
    result_epoch_2.bias,
    ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    "target",
)
print(f"Inference Epoch 2 Accuracy: {result_inference.Accuracy()}")
print(f"Inference Epoch 2 MSE: {result_inference.MSE()}")
print("========================")

# EPOCH 3
print("\n========================")
print("Training Epoch 3")
result_epoch_3 = classifier.fit(
    train_df,
    LEARNING_RATE,
    INIT_WEIGHT,
    INIT_BIAS,
    ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    "target",
    3,
)

print(f"Epoch 3 Weight: {result_epoch_3.GetWeight()}")
print(f"Epoch 3 Bias: {result_epoch_3.GetBias()}")
print(f"Epoch 3 Training MSE: {result_epoch_3.GetMSEPerEpoch()}")
print(f"Epoch 3 Training Accuracy: {result_epoch_3.GetAccuracyPerEpoch()}")
print("-------------------------")

# Inference EPOCH 3
print("Inference Epoch 3")
result_inference = classifier.inference(
    test_df,
    result_epoch_3.weight,
    result_epoch_3.bias,
    ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    "target",
)
print(f"Inference Epoch 3 Accuracy: {result_inference.Accuracy()}")
print(f"Inference Epoch 3 MSE: {result_inference.MSE()}")
print("========================")

# EPOCH 4
print("\n========================")
print("Training Epoch 4")
result_epoch_4 = classifier.fit(
    train_df,
    LEARNING_RATE,
    INIT_WEIGHT,
    INIT_BIAS,
    ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    "target",
    4,
)

print(f"Epoch 4 Weight: {result_epoch_4.GetWeight()}")
print(f"Epoch 4 Bias: {result_epoch_4.GetBias()}")
print(f"Epoch 4 Training MSE: {result_epoch_4.GetMSEPerEpoch()}")
print(f"Epoch 4 Training Accuracy: {result_epoch_4.GetAccuracyPerEpoch()}")
print("-------------------------")

# Inference EPOCH 4
print("Inference Epoch 4")
result_inference = classifier.inference(
    test_df,
    result_epoch_4.weight,
    result_epoch_4.bias,
    ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    "target",
)
print(f"Inference Epoch 4 Accuracy: {result_inference.Accuracy()}")
print(f"Inference Epoch 4 MSE: {result_inference.MSE()}")
print("========================")

# EPOCH 5
print("\n========================")
print("Training Epoch 5")
result_epoch_5 = classifier.fit(
    train_df,
    LEARNING_RATE,
    INIT_WEIGHT,
    INIT_BIAS,
    ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    "target",
    5,
)

print(f"Epoch 5 Weight: {result_epoch_5.GetWeight()}")
print(f"Epoch 5 Bias: {result_epoch_5.GetBias()}")
print(f"Epoch 5 Training MSE: {result_epoch_5.GetMSEPerEpoch()}")
print(f"Epoch 5 Training Accuracy: {result_epoch_5.GetAccuracyPerEpoch()}")
print("-------------------------")

# Inference EPOCH 5
print("Inference Epoch 5")
result_inference = classifier.inference(
    test_df,
    result_epoch_5.weight,
    result_epoch_5.bias,
    ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    "target",
)
print(f"Inference Epoch 5 Accuracy: {result_inference.Accuracy()}")
print(f"Inference Epoch 5 MSE: {result_inference.MSE()}")
print("========================")
