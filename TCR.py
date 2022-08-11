import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from time import time
import math
import DL
# metrics are used to find accuracy or error
from sklearn import metrics

#####################
BATCH_SIZE = 64  # Choose batch size for deep learning training process
NUM_PROCESS = 2  # Choose how many CPU cores are used for training neural networks
INCLUDE_DL = False  # Choose whether to run DL benchmark. This requires PyTorch
# Currently also support attention-based transformer, simply add 'transformer' to this list.
DL_Models = ['CNN', 'LSTM']
#####################

data_source = 'TCR'
task_num = 5
subject_id_start = 1
subject_id_end = 17  # tcr's max 17; rwt's max 14
tcr_subject = [7, 112, 113, 121, 75, 107, 79, 82,
               118, 76, 115, 117, 119, 120, 105, 78, 124]
rwt_subject = [7, 112, 113, 114, 75, 107, 79, 82, 118, 76, 115, 117, 119, 120]
subject = None

if data_source == 'TCR':
    subject = tcr_subject
else:
    subject = rwt_subject


ground_truth = 1
distribution_list = {}
while ground_truth <= task_num:
    distribution_list[ground_truth] = 0
    ground_truth += 1

first_chopped_off = 600 * 0.3
last_chopped_off = 600 * 0

Random_Forest_predicted_y = []
RBF_SVM_predicted_y = []

XGBoost = {}
GradientBoost = {}
NearestNeighbor = {}
AdaBoost = {}
RandomForest = {}
LinearSVM = {}
RBFSVM = {}
DecisionTree = {}
RUSBoost = {}
LDA = {}
sLDA = {}
CNN = {}
LSTM = {}

# records the number of sessions and folds left for each subjects
subject_preprocess_record = {}

subject_unknown_percentage = {}

# Records the standard deviation of each classifiers
standard_deviation_record = {}

# records the time it takes for each classifier to execute the 7-fold cross-validation
time_classifier = {}
# define models to train
# define models to train
names = [
    # 'XGBoost',
    # 'GradientBoosting',
    # 'LDA',
    # 'Nearest Neighbors',
    # 'AdaBoostClassifier',
    # 'RandomForest',
    # "Linear SVM",
    # "RBF SVM",
    # "Decision Tree",
    # "Shrinkage LDA",
]

# build classifiers
classifiers = [
    # XGBClassifier(eval_metric='mlogloss'),
    # GradientBoostingClassifier(),
    # LinearDiscriminantAnalysis(),
    # KNeighborsClassifier(n_neighbors=5),
    # AdaBoostClassifier(),
    # RandomForestClassifier(
    #     n_estimators=300, max_features="sqrt", oob_score=True),
    # SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    # DecisionTreeClassifier(),
    # LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
]

dicts_records = [
    # XGBoost,
    # GradientBoost,
    # LDA,
    # NearestNeighbor,
    # AdaBoost,
    # RandomForest,
    # LinearSVM,
    # RBFSVM,
    # DecisionTree,
    # sLDA
]


def check_removed_index(name, removed_dict, index_to_be_removed, lowerBound, upperBound):
    if name not in removed_dict:
        removed_dict[name] = 0
    lst = [i for i in index_to_be_removed if i >=
           lowerBound and i < upperBound]
    removed_dict[name] = removed_dict[name] + len(lst)


def calculate_accuracy(y_actual, y_predict):
    count = 0
    for i in range(len(y_actual)):
        if y_actual[i] == y_predict[i]:
            count = count + 1
    return count / float(len(y_actual))


def check_plateau(dataFrame, current_index):
    for i in range(current_index + 1, current_index + 14):
        if i >= len(dataFrame):
            return True
        lst = move_data.iloc[current_index, :].tolist()
        if sum(lst) != 0:
            return False
    return True

# Calculate the sample standard deviation given a list of accuracies
def calculate_sd(accuracies, mean):
    sum_of_accuracies = 0
    for acc in accuracies:
        sum_of_accuracies += pow(acc-mean, 2)
    sum_of_accuracies /= len(accuracies) - 1
    return math.sqrt(sum_of_accuracies)

# Truncate each folds to make sure the length of each folds is the multiple of the BATCH_SIZE
def folds_truncate(folds):
    folds_DL = []
    for ele in folds:
        ele = ele.reset_index(drop=True)
        print("The shape of each fold in DL model before is", ele.shape)
        num_of_batches = len(ele.index) // BATCH_SIZE
        fold_truncate_after = BATCH_SIZE * num_of_batches - 1
        print("fold_truncate_after is", fold_truncate_after)
        ele = ele.truncate(before=0, after=fold_truncate_after)
        folds_DL.append(ele)
        print("The length of each fold in DL model after is", ele.shape)
        print("=" * 8)
    return folds_DL

def label_distribution(folds : list, subject_id):
    for fold in folds:
        for index, row in fold.iterrows():
            ground_truth = row["ground_truth"]
            distribution_list[ground_truth] += 1
    print("label saved for subeject", subject_id)

def label_distribution_save(distribution_list):
    distribution_percentage = []
    total = sum(distribution_list.values())
    for key in distribution_list:
        percentage = distribution_list[key] / total
        distribution_percentage.append(round(percentage,3))
    data = {'Distribution  Number': distribution_list.values(), 'Distribution Percentage': distribution_percentage}
    df = pd.DataFrame(data, index=distribution_list.keys())
    df.index.name = 'Task Number'
    df.to_csv(f"output/{data_source}/{data_source}_label_distribution.csv")
    print(df)

print(len(subject))

for i in range(subject_id_start, subject_id_end + 1):
    subject_id = i
    print()
    print("checking subject", subject_id)
    print()
    frame = []
    for session in range(1, 7):
        data_one = pd.read_csv(
             f"data/{data_source}/{data_source}_processed/{data_source.lower()}_subject_{str(subject_id)}_session_{str(session)}.csv",
            header=None)
        zeros = [0] * 20

        if len(data_one) <= 3000:
            data_one.loc[len(data_one)] = zeros
        data_one = data_one.iloc[0:3000]
        temp_frame = []
        for i in range(5):
            temp = data_one.iloc[int(
                i * 600 + first_chopped_off): int((i + 1) * 600 - last_chopped_off)]
            temp_frame.append(temp)
        data_one = pd.concat(temp_frame)
        frame.append(data_one)
    data = pd.concat(frame)
    data.reset_index(drop=True, inplace=True)

    # check plateau (noise)
    index_to_be_removed = []
    for session in range(0, 6):
        temp_data = data.iloc[session * 2100: (session + 1) * 2100]
        for move in range(0, 5):
            move_data = temp_data.iloc[move * 420: (move + 1) * 420]
            for i in range(len(move_data)):
                lst = move_data.iloc[i, :].tolist()
                if sum(lst) == 0:
                    index_to_be_removed.append(session * 2100 + move * 420 + i)
    print("number of rows to be removed is", len(index_to_be_removed))

    # add labels
    ones = [1] * int(600 * 0.7)
    twos = [2] * int(600 * 0.7)
    threes = [3] * int(600 * 0.7)
    fours = [4] * int(600 * 0.7)
    fives = [5] * int(600 * 0.7)
    len(fives)
    session1 = ones + twos + threes + fours + fives
    session2 = fours + ones + twos + threes + fives
    session3 = ones + fours + threes + twos + fives
    session4 = ones + twos + threes + fours + fives
    session5 = twos + ones + threes + fives + fours
    session6 = ones + twos + fours + threes + fives
    session_all = session1 + session2 + session3 + session4 + session5 + session6
    data["ground_truth"] = session_all

    # check if the subject should be kept
    percentage_removed_total = (
        int(18000 * 0.7) - len(index_to_be_removed)) / 18000.0
    print("percentage of data left for subject",
          subject_id, "is", percentage_removed_total)
    if percentage_removed_total < 0.35:
        print("the subject", subject_id, "should be removed and will be ignored")
        continue
    if str(subject_id) not in subject_unknown_percentage:
        subject_unknown_percentage[str(subject_id)] = {}
    subject_unknown_percentage[str(
        subject_id)]["known"] = percentage_removed_total
    subject_preprocess_record[str(subject_id)] = {}

    # checks each six session:
    print("check session for subject", subject_id)
    session_list = [i for i in range(0, 6)]
    for session in range(0, 6):
        session_lowerbound = 2100 * session
        session_upperbound = 2100 * (session + 1)
        to_be_removed = [i for i in index_to_be_removed if i >=
                         session_lowerbound and i < session_upperbound]
        print("Number of rows to be removed for session",
              (session + 1), "is", len(to_be_removed))
        percentage_remained = (2100 - (len(to_be_removed))) / 3000.0
        print("percent of rows left in sesssion",
              (session + 1), "is", percentage_remained)
        if percentage_remained < 0.35:
            session_list.remove(session)
            print("session", session, " should be removed and will be ignored")
            print()
        print()
    if len(session_list) == 0:
        print("all sessions are ignored. Continue to next person")
        continue

    subject_preprocess_record[str(
        subject_id)]["session_remained"] = len(session_list)

    # cut 7 folds
    test1data = []
    test2data = []
    test3data = []
    test4data = []
    test5data = []
    test6data = []
    test7data = []
    removed_dict = {}
    fold_names = ["fold1", "fold2", "fold3",
                  "fold4", "fold5", "fold6", "fold7"]
    # for each move (420 lines), split the data into seven folds
    # at the same time, record the number of lines being that would be omited
    # remove folds that have less than 33.33% data remained
    # each fold should have at most 60 * 30 = 1800 (originally 2571.4)
    move_lst = []
    for ele in session_list:
        temp = [i for i in range(ele * 5, (ele + 1) * 5)]
        move_lst.extend(temp)
    for i in move_lst:
        lowerBound = i * 420
        test1data.append(data.iloc[lowerBound: lowerBound + 60])
        check_removed_index("fold1", removed_dict,
                            index_to_be_removed, lowerBound, lowerBound + 60)
        test2data.append(data.iloc[lowerBound + 60: lowerBound + 120])
        check_removed_index(
            "fold2", removed_dict, index_to_be_removed, lowerBound + 60, lowerBound + 120)
        test3data.append(data.iloc[lowerBound + 120: lowerBound + 180])
        check_removed_index(
            "fold3", removed_dict, index_to_be_removed, lowerBound + 120, lowerBound + 180)
        test4data.append(data.iloc[lowerBound + 180: lowerBound + 240])
        check_removed_index(
            "fold4", removed_dict, index_to_be_removed, lowerBound + 180, lowerBound + 240)
        test5data.append(data.iloc[lowerBound + 240: lowerBound + 300])
        check_removed_index(
            "fold5", removed_dict, index_to_be_removed, lowerBound + 240, lowerBound + 300)
        test6data.append(data.iloc[lowerBound + 300: lowerBound + 360])
        check_removed_index(
            "fold6", removed_dict, index_to_be_removed, lowerBound + 300, lowerBound + 360)
        test7data.append(data.iloc[lowerBound + 360: lowerBound + 420])
        check_removed_index(
            "fold7", removed_dict, index_to_be_removed, lowerBound + 360, lowerBound + 420)

    folds_list = [test1data, test2data, test3data,
                  test4data, test5data, test6data, test7data]
    # check folds percentages
    for name in fold_names:
        removed_num = removed_dict[name]
        remained_percentage = (((2100 * len(session_list)) / 7.0) -
                               removed_num) / ((3000 * len(session_list)) / 7.0)
        print()
        print("the " + name + " has", remained_percentage, "left")
        print()
        if remained_percentage < 0.35:
            print(name + " should be removed for subject", subject_id)
            idx = fold_names.index(name)
            print("the index to be removed is", idx)
            del folds_list[idx]
            fold_names.remove(name)
    folds = []
    print("Removed dict is", removed_dict)
    print(sum(removed_dict.values()))
    for fold in folds_list:
        data = pd.concat(fold)
        index = data.index.tolist()
        index_remove = index_to_be_removed.copy()
        index_remove = [i for i in index_remove if i in index]
        data = data.drop(labels=index_remove, axis=0)
        print(data.shape)
        folds.append(data)
    if len(folds) <= 1:
        print("all folds are ignored. Continue to next person")
        continue

    subject_preprocess_record[str(subject_id)]["folds_remained"] = len(folds)

    label_distribution(folds, subject_id)

    models = zip(names, classifiers, dicts_records)
    for name, classifier, dicts_record in models:
        accuracy = 0
        t0 = time()
        accuracy_list = []
        for i in range(len(folds_list)):
            folds.append(folds.pop(0))
            data = pd.concat(folds[:-1])
            index = data.index.tolist()
            index_remove = index_to_be_removed.copy()
            index_remove = [i for i in index_remove if i in index]
            data = data.drop(labels=index_remove, axis=0)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            clf = classifier
            clf.fit(X, y)
            data_test = folds[-1]
            X_test = data_test.iloc[:, :-1]
            y_test = data_test.iloc[:, -1]
            y_predict = []
            y_predict = clf.predict(X_test)
            if name == 'RandomForest':
                Random_Forest_predicted_y.extend(y_predict)
            if name == 'RBF SVM':
                RBF_SVM_predicted_y.extend(y_predict)
            accuracy_list.append(metrics.accuracy_score(y_test, y_predict))
            accuracy += metrics.accuracy_score(y_test, y_predict)
        t1 = time()
        time_elapsed = t1 - t0
        print()
        print("The time it takes to run " + name + " is", time_elapsed)
        if name not in time_classifier:
            time_classifier[name] = 0
        time_classifier[name] = time_classifier[name] + time_elapsed
        accuracy = accuracy / float(len(folds_list))
        dicts_record[str(subject_id)] = accuracy
        standard_deviation_subject = calculate_sd(accuracy_list, accuracy)
        standard_deviation_record[str(subject_id)] = standard_deviation_subject
        print("The accuracy of subject", subject_id,
              "is", accuracy, "with the model " + name)
        print("The standard deviation of subject", subject_id,
              "is", standard_deviation_subject, "with the model " + name)

    if INCLUDE_DL:
        folds = folds_truncate(folds)
        for name in DL_Models:
            accs = []
            t0 = time()
            execute_counter = 1
            for i in range(len(folds_list)):

                folds.append(folds.pop(0))
                print("inside DL model, length of first fold", folds[0].shape)

                data = pd.concat(folds[:-1])
                index = data.index.tolist()
                index_remove = index_to_be_removed.copy()
                index_remove = [i for i in index_remove if i in index]

                data = data.drop(labels=index_remove, axis=0)
                npdata = np.array(data)
                tcr = DL.EEGCogNet_DL(npdata)
                train_loader = DataLoader(
                    dataset=tcr, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_PROCESS)

                model = DL.DL_Model(name, BATCH_SIZE)
                model.fit(train_loader)

                data_test = folds[-1]
                testdata = np.array(data_test)
                tcr_test = DL.EEGCogNet_DL(testdata)
                test_loader = DataLoader(
                    dataset=tcr_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_PROCESS)
                cur_accuracy = DL.test_loop(test_loader, model.net)

                accs.append(cur_accuracy)

            t1 = time()
            time_elapsed = t1 - t0
            print()
            time_classifier[name] = time_elapsed
            print("The time it takes to run " + name + " is", time_elapsed)
            average_accuracy = sum(accs) / (float)(len(folds_list))
            if name == "CNN":
                CNN[str(subject_id)] = average_accuracy
            elif name == "LSTM":
                LSTM[str(subject_id)] = average_accuracy
            sd_dl = calculate_sd(accs, average_accuracy)
            print("The accuracy of subject", subject_id, "is",
                  average_accuracy, "with the model " + name)
            print("The sstandard deviation of subject", subject_id, "is",
                  round(sd_dl, 5), "with the model " + name)


label_distribution_save(distribution_list)

print("Dicts_order is:")
print(dicts_records)
print("="*10)
for name, dicts_record in zip(names, dicts_records):
    print(name)
    print(dicts_record)
    print()

total = 0
best_classifier_name = ""
dict_sum_recorder = {}
for name, dicts_record in zip(names, dicts_records):
    cur = sum(dicts_record.values())
    dict_sum_recorder[name] = cur
    if cur > total:
        total = cur
        best_classifier_name = name
best_classifier = classifiers[names.index(best_classifier_name)]
best_classifier_dict = dicts_records[names.index(best_classifier_name)]
print("The best classifier is: " + best_classifier_name)
print("the dictionary for the best classifier is: ")
print(best_classifier_dict)
print()

dict_sum_recorder = dict(
    sorted(dict_sum_recorder.items(), key=lambda item: -item[1]))
print("The dic_sum_recorder is")
print(dict_sum_recorder)
print()
classifier_order = list(dict_sum_recorder.keys())
print("the order of the classifier is: ")
print(classifier_order)
print()
best_classifier_dict_sorted = dict(
    sorted(best_classifier_dict.items(), key=lambda item: -item[1]))
# The x axis of the plot
subject_id_order = list(best_classifier_dict_sorted.keys())
print("best_classifier_dict_sorted is: ")
print(best_classifier_dict_sorted)
print()
print("The order of the subject id is")
print(subject_id_order)
print()

result_y_res = []  # each element follows the order of the classifier_order
for i in range(len(classifier_order)):
    temp_lst = []
    classifier = classifier_order[i]
    for subject in subject_id_order:
        idx = names.index(classifier)
        temp_lst.append(dicts_records[idx][subject])
    result_y_res.append(temp_lst)

print(result_y_res)

subject_id_order.reverse()
x_axis = subject_id_order

print(x_axis)


avg_accuracy = []
time = []
sd_classifier = []
name_list = []
print("The time_classifier is", time_classifier)
print("The dict_sum_recorder is", dict_sum_recorder)
print("Number of subjects is", len(subject_id_order))
for ele in dict_sum_recorder:
    pos_of_ele = names.index(ele)
    list_of_accuracies_of_ele = dicts_records[pos_of_ele].values()
    name_list.append(ele)
    temp_avg = dict_sum_recorder[ele] / float(len(subject_id_order))
    sd_classifier.append(round(calculate_sd(list_of_accuracies_of_ele, temp_avg),5))
    avg_accuracy.append(round(temp_avg, 2))
    temp_time = time_classifier[ele] / float(len(subject_id_order))
    time.append(round(temp_time, 1))
print()
print("avg accuracy", avg_accuracy)
print("time", time)
print("name order", name_list)
print("Standard Deviation of classifiers", sd_classifier)
if INCLUDE_DL:
    temp_avg_CNN = sum(CNN.values()) / float(len(CNN))
    time_CNN = time_classifier["CNN"]
    sd_CNN = calculate_sd(CNN.values(), temp_avg_CNN)
    temp_avg_LSTM = sum(LSTM.values()) / float(len(LSTM))
    time_LSTM = time_classifier["LSTM"]
    sd_LSTM = calculate_sd(LSTM.values(), temp_avg_LSTM)
    avg_accuracy.append(temp_avg_CNN)
    avg_accuracy.append(temp_avg_LSTM)
    time.append(time_CNN)
    time.append(time_LSTM)
    sd_classifier.append(sd_CNN)
    sd_classifier.append(sd_LSTM)

data = {'Average Accuracy': avg_accuracy, 'Avg code runtime(s)': time, 'Standard Deviation': sd_classifier}
# Creates pandas DataFrame.
if INCLUDE_DL:
    df = pd.DataFrame(data, index=name_list + DL_Models)
else:
    df = pd.DataFrame(data, index=name_list)

df = pd.DataFrame(data, index=name_list)
df.to_csv(f"output/{data_source}/accuracy_runtime_classifier.csv")
print(df)

plt.figure(figsize=(10, 5))
fig, ax = plt.subplots()
for i in range(len(result_y_res)):
    result_y_res[i].reverse()
    y = result_y_res[i]
    label_name = classifier_order[i]
    temp_avg = dict_sum_recorder[label_name] / float(len(subject_id_order))
    temp_avg = round(temp_avg, 2)
    if label_name == "Shrinkage LDA":
        label_name = "sLDA"
    if label_name == "Nearest Neighbors":
        label_name = "KNN"
    if label_name == "AdaBoostClassifier":
        label_name = "AdaBoost"
    ax.plot(x_axis, y, marker='D', label=label_name + "(" + str(temp_avg)+")")

ax.set_position([0.1, 0.5, 1.2, 1.0])
ax.legend(loc='upper left')
plt.axhline(y=0.2, color='r', linestyle=':')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Subject ID orderd by ' + best_classifier_name, fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.savefig(f"output/{data_source}/algorithm_comparison_each_subject.jpg", bbox_inches='tight', dpi=2000)
plt.show()
