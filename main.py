import pandas as pd
import default_classifier
import matplotlib.pyplot as plt

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

train_mse_list = []
test_mse_list = []
train_acc_list = []
test_acc_list = []

feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

for epoch in range(1, EPOCH + 1):
    print("\n========================")
    print(f"Training Epoch {epoch}")

    result = classifier.fit(
        train_df,
        LEARNING_RATE,
        INIT_WEIGHT,
        INIT_BIAS,
        feature_cols,
        "target",
        epoch,
    )

    # training metric
    train_mse = result.mse_last_epoch
    train_acc = result.accuracy_last_epoch

    train_mse_list.append(train_mse)
    train_acc_list.append(train_acc)

    print(f"Epoch {epoch} Training MSE: {train_mse}")
    print(f"Epoch {epoch} Training Accuracy: {train_acc}")

    # inference
    result_inference = classifier.inference(
        test_df,
        result.weight,
        result.bias,
        feature_cols,
        "target",
    )

    test_mse = result_inference.MSE()
    test_acc = result_inference.Accuracy()

    test_mse_list.append(test_mse)
    test_acc_list.append(test_acc)

    print(f"Inference Epoch {epoch} Accuracy: {test_acc}")
    print(f"Inference Epoch {epoch} MSE: {test_mse}")
    print("========================")


train_mse_list = [float(x) for x in train_mse_list]
test_mse_list = [float(x) for x in test_mse_list]
train_acc_list = [float(x) for x in train_acc_list]
test_acc_list = [float(x) for x in test_acc_list]
epochs = list(range(1, len(train_mse_list) + 1))

# Plot Error
plt.figure()
plt.plot(epochs, train_mse_list)
plt.plot(epochs, test_mse_list)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training vs Testing Error")
plt.legend(["Training", "Testing"])
plt.xticks(epochs)
plt.grid(True)
plt.savefig("results/training_vs_testing_mse.png", dpi=300)
plt.show()

# Plot Accuracy
plt.figure()
plt.plot(epochs, train_acc_list)
plt.plot(epochs, test_acc_list)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy")
plt.legend(["Training", "Testing"])
plt.xticks(epochs)
plt.grid(True)
plt.savefig("results/training_vs_testing_accuracy.png", dpi=300)
plt.show()
