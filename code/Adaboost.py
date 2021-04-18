from CS6735 import *

print("[INFO] loading and preparing the first dataset, (canc)...")
names = ["feature_" + str(i) for i in range(9)] + ["target"]
a = "Accuracy Adaboost on cancer dataset"
canc = (
    pd.read_csv(
        files.canc_dataset_file,
        header=0,
        names=names,
        index_col=0,
        parse_dates=True,
        squeeze=True,
    )
    .replace("?", np.NaN)
    .dropna()
    .reset_index()
)
canc.drop(columns=["index"], inplace=True)

canc.replace(
    {
        "target": {
            4.0: 1,
            2.0: -1,
        },
    },
    inplace=True,
)

for i in range(9):
    canc[names[i]] = discr(canc[names[i]].values.astype(np.float), 5)

dataset = np.array(canc.values).astype(np.float).tolist()
acc = list()
for _ in range(10):
    adaboost = AB(n_folds=5, dataset=dataset, n_clf=5)
    accuracy = adaboost.fit()
    acc.append(sum(accuracy) / len(accuracy))

acc1 = copy.deepcopy(acc)
acc1.append(np.average(acc))
acc1.append(np.std(acc))
print(a + " (Average): {:0.2f}%".format((sum(acc) / len(acc))))
index = (
    ["The 1st run", "The 2nd run", "The 3rd run"]
    + ["The " + str(i + 4) + "th run" for i in range(7)]
    + ["Average accuracy:", "Standard deviation:"]
)
df = pd.DataFrame(np.around(acc1, 2), columns=[a], index=index)

print("[INFO] Saving Results on file...")
with open(os.path.join(files.tbl_dir, a + ".tex"), "w") as tf:
    tf.write(df.to_latex(index=True))


#############################
#############################
###########  mush  ##########
#############################
#############################
print("[INFO] loading and preparing the fifth dataset, (mush)...")
names = ["target"] + ["feature_" + str(i) for i in range(22)]
mush = (
    pd.read_csv(
        files.mush_dataset_file,
        header=0,
        names=names,
        index_col=0,
        parse_dates=True,
        squeeze=True,
    )
    .replace("?", np.NaN)
    .dropna()
).reset_index()

# print(set(mush["feature_19"]))
# print(mush.head())

mush.replace(
    {
        "target": {"e": 1, "p": -1},
        "feature_0": {"x": 1, "c": 2, "f": 3, "k": 4, "s": 5, "b": 6},
        "feature_1": {"s": 1, "g": 2, "y": 3, "f": 4},
        "feature_2": {
            "r": 1,
            "u": 2,
            "y": 3,
            "p": 4,
            "w": 5,
            "c": 6,
            "e": 7,
            "n": 8,
            "g": 9,
            "b": 10,
        },
        "feature_3": {"f": 1, "t": 2},
        "feature_4": {
            "l": 1,
            "f": 2,
            "y": 3,
            "m": 4,
            "c": 5,
            "p": 6,
            "a": 7,
            "s": 8,
            "n": 9,
        },
        "feature_5": {"f": 1, "a": 2},
        "feature_6": {"w": 1, "c": 2},
        "feature_7": {"b": 1, "n": 2},
        "feature_8": {
            "w": 1,
            "y": 2,
            "e": 3,
            "n": 4,
            "k": 5,
            "g": 6,
            "u": 7,
            "b": 8,
            "r": 9,
            "p": 10,
            "h": 11,
            "o": 12,
        },
        "feature_9": {"e": 1, "t": 2},
        "feature_10": {"b": 1, "c": 2, "e": 3, "r": 4},  ##
        "feature_11": {"f": 1, "y": 2, "s": 3, "k": 4},
        "feature_12": {"f": 1, "y": 2, "s": 3, "k": 4},
        "feature_13": {
            "w": 1,
            "y": 2,
            "p": 3,
            "o": 4,
            "n": 5,
            "e": 6,
            "c": 7,
            "g": 8,
            "b": 9,
        },
        "feature_14": {
            "w": 1,
            "y": 2,
            "p": 3,
            "o": 4,
            "n": 5,
            "e": 6,
            "c": 7,
            "g": 8,
            "b": 9,
        },
        "feature_15": {"p": 1},
        "feature_16": {"o": 1, "w": 2, "y": 3, "n": 4},
        "feature_17": {"o": 1, "n": 2, "t": 3},
        "feature_18": {"e": 1, "f": 2, "l": 3, "p": 4, "n": 5},
        "feature_19": {
            "y": 1,
            "b": 2,
            "n": 3,
            "k": 4,
            "w": 5,
            "r": 6,
            "o": 7,
            "h": 8,
            "u": 9,
        },
        "feature_20": {
            "y": 1,
            "n": 2,
            "v": 3,
            "c": 4,
            "a": 5,
            "s": 6,
        },
        "feature_21": {
            "l": 1,
            "w": 2,
            "d": 3,
            "p": 4,
            "g": 5,
            "m": 6,
            "u": 7,
        },
    },
    inplace=True,
)
mush["target_1"] = mush["target"]
mush.drop(columns=["target"], inplace=True)
# print(mush.head())
dataset = np.array(mush.values).astype(np.float).tolist()
acc = list()
for _ in range(10):
    adaboost = AB(n_folds=5, dataset=dataset, n_clf=5)
    accuracy = adaboost.fit()
    acc.append(sum(accuracy) / len(accuracy))

acc1 = copy.deepcopy(acc)
acc1.append(np.average(acc))
acc1.append(np.std(acc))
a = "Accuracy Adaboost on mushroom dataset"

print(a + " (Average): {:0.2f}%".format((sum(acc) / len(acc))))
index = (
    ["The 1st run", "The 2nd run", "The 3rd run"]
    + ["The " + str(i + 4) + "th run" for i in range(7)]
    + ["Average accuracy:", "Standard deviation:"]
)
df = pd.DataFrame(np.around(acc1, 2), columns=[a], index=index)

print("[INFO] Saving Results on file...")
with open(os.path.join(files.tbl_dir, a + ".tex"), "w") as tf:
    tf.write(df.to_latex(index=True))
