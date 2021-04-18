from CS6735 import *

print("[INFO] loading and preparing the first dataset, (canc)...")
names = ["feature_" + str(i) for i in range(9)] + ["target"]
a = "Accuracy ID3 on cancer dataset"
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

for i in range(9):
    canc[names[i]] = discr(canc[names[i]].values.astype(np.float), 5)

dataset = np.array(canc.values).astype(np.float).tolist()
acc = list()
for _ in range(10):
    id3 = ID3(n_folds=5, dataset=dataset, names=names)
    accuracy = id3.fit()
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


mush.replace(
    {
        "target": {"e": 1, "p": 2},
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

dataset = np.array(mush.values).astype(np.float).tolist()
acc = list()
for _ in range(10):
    id3 = ID3(n_folds=5, dataset=dataset, names=names)
    accuracy = id3.fit()
    acc.append(sum(accuracy) / len(accuracy))

acc1 = copy.deepcopy(acc)
acc1.append(np.average(acc))
acc1.append(np.std(acc))
a = "Accuracy ID3 on mushroom dataset"

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
############ cars  ##########
#############################
#############################
print("[INFO] loading and preparing the Second dataset, (cars)...")
names = ["feature_" + str(i) for i in range(6)] + ["target"]

cars = (
    pd.read_csv(
        files.cars_dataset_file,
        header=0,
        names=names,
        index_col=0,
        parse_dates=True,
        squeeze=True,
    ).dropna()
).reset_index()

cars.replace(
    {
        "feature_0": {"vhigh": 4, "med": 2, "high": 3, "low": 1},
        "feature_1": {"vhigh": 4, "med": 2, "high": 3, "low": 1},
        "feature_2": {"2": 1, "3": 2, "4": 3, "5more": 4},
        "feature_3": {"2": 1, "4": 2, "more": 3},
        "feature_4": {"small": 1, "med": 2, "big": 3},
        "feature_5": {"med": 2, "high": 3, "low": 1},
        "target": {"good": 1, "vgood": 2, "acc": 3, "unacc": 4},
    },
    inplace=True,
)

dataset = np.array(cars.values).astype(np.float).tolist()
acc = list()
for _ in range(10):
    id3 = ID3(n_folds=5, dataset=dataset, names=names)
    accuracy = id3.fit()
    acc.append(sum(accuracy) / len(accuracy))

acc1 = copy.deepcopy(acc)
acc1.append(np.average(acc))
acc1.append(np.std(acc))
a = "Accuracy ID3 on cars dataset"

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
############ ecol  ##########
#############################
#############################
print("[INFO] loading and preparing the third dataset, (ecol)...")
names = ["feature_" + str(i) for i in range(8)] + ["target"]

ecol = (
    pd.read_csv(
        files.ecol_dataset_file,
        header=0,
        names=names,
        index_col=0,
        parse_dates=True,
        squeeze=True,
    ).dropna()
).reset_index()

# print(set(ecol["target"]))
ecol.replace(
    {
        "target": {
            "im": 1,
            "om": 2,
            "imL": 3,
            "pp": 4,
            "imU": 5,
            "cp": 6,
            "imS": 7,
            "omL": 8,
        },
    },
    inplace=True,
)
ecol.drop(columns=["feature_0"], inplace=True)
# print(ecol.head(10))
for i in range(7):
    ecol[names[i + 1]] = discr(ecol[names[i + 1]].values.astype(np.float), 5)


# 1 / 0
dataset = np.array(ecol.values).astype(np.float).tolist()


acc = list()
for _ in range(10):
    id3 = ID3(n_folds=5, dataset=dataset, names=ecol.columns.to_list())
    accuracy = id3.fit()
    acc.append(sum(accuracy) / len(accuracy))

acc1 = copy.deepcopy(acc)
acc1.append(np.average(acc))
acc1.append(np.std(acc))
a = "Accuracy ID3 on ecol dataset"

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
############ lett  ##########
#############################
#############################
print("[INFO] loading and preparing the fourth dataset, (lett)...")
names = ["target"] + ["feature_" + str(i) for i in range(16)]
lett = (
    pd.read_csv(
        files.lett_dataset_file,
        header=0,
        names=names,
        index_col=0,
        parse_dates=True,
        squeeze=True,
    ).dropna()
).reset_index()
# print(set(lett["target"]))
lett.replace(
    {
        "target": {
            "Y": 1,
            "H": 2,
            "C": 3,
            "F": 4,
            "S": 5,
            "I": 6,
            "J": 7,
            "M": 8,
            "A": 9,
            "L": 10,
            "P": 11,
            "N": 12,
            "Q": 13,
            "X": 14,
            "V": 15,
            "T": 16,
            "D": 17,
            "U": 18,
            "K": 19,
            "Z": 20,
            "R": 21,
            "W": 22,
            "O": 23,
            "B": 24,
            "G": 25,
            "E": 26,
        }
    },
    inplace=True,
)
lett.drop(columns=["feature_0"], inplace=True)
lett["target_1"] = lett["target"]
lett.drop(columns=["target"], inplace=True)

for i in range(15):
    lett[names[i + 2]] = discr(lett[names[i + 2]].values.astype(np.float), 5)


dataset = np.array(lett.values).astype(np.float).tolist()

acc = list()
for _ in range(10):
    id3 = ID3(n_folds=5, dataset=dataset, names=lett.columns.to_list())
    accuracy = id3.fit()
    acc.append(sum(accuracy) / len(accuracy))

acc1 = copy.deepcopy(acc)
acc1.append(np.average(acc))
acc1.append(np.std(acc))
a = "Accuracy ID3 on letter dataset"

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
