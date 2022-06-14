import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from time import time
# metrics are used to find accuracy or error
from sklearn import metrics

data_source = 'tcr'
subject_id_start = 1
subject_id_end = 17 # tcr's max 17; rwt's max 14
data_status = 'old_'
tcr_subject = [7, 112, 113, 121, 75, 107, 79, 82, 118, 76, 115, 117, 119, 120, 105, 78, 124]
rwt_subject = [7, 112, 113, 114, 75, 107, 79, 82, 118, 76, 115, 117, 119, 120]
new_tcr_trial = [123, 124]
subject = None
if data_status == "old_":
    if data_source == 'tcr':
        subject = tcr_subject
    else:
        subject = rwt_subject
else:
    subject = new_tcr_trial
first_chopped_off = 600 * 0.3
last_chopped_off = 600 * 0

Random_Forest_predicted_y = []
RBF_SVM_predicted_y = []


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

subject_preprocess_record = {} # records the number of sessions and folds left for each subjects

subject_prediction = {}

subject_unknown_percentage = {}

time_classifier = {} # records the time it takes for each classifier to execute the 7-fold cross-validation
# define models to train
# define models to train
names = [
    'GradientBoosting',
    'LDA',
    'Nearest Neighbors',
    'AdaBoostClassifier',
    'RandomForest',
    "Linear SVM",
    "RBF SVM",
    "Decision Tree",
    "Shrinkage LDA",
]

# build classifiers
classifiers = [
    GradientBoostingClassifier(),
    LinearDiscriminantAnalysis(),
    KNeighborsClassifier(n_neighbors=5),
    AdaBoostClassifier(),
    RandomForestClassifier(n_estimators=300, max_features="sqrt", oob_score=True),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(),
    LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
]

dicts_records = [
                 GradientBoost,
                 LDA,
                 NearestNeighbor,
                 AdaBoost,
                 RandomForest,
                 LinearSVM,
                 RBFSVM,
                 DecisionTree,
                 sLDA
                ]


def check_removed_index(name, removed_dict, index_to_be_removed, lowerBound, upperBound):
    if name not in removed_dict:
        removed_dict[name] = 0
    lst = [i for i in index_to_be_removed if i >= lowerBound and i < upperBound]
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

print(len(subject))

for i in range(subject_id_start, subject_id_end + 1):
    subject_id = i
    print()
    print("checking subject", subject_id)
    print()
    frame = []
    for session in range(1, 7):
        data_one = pd.read_csv(
            'data_preprocess/' + data_status + data_source + '_plateau_removed_data/' + data_source + "_subject_" + str(
                subject_id) + "_session_" + str(session) + ".csv",
            header=None)
        zeros = [0] * 20

        if len(data_one) <= 3000:
            data_one.loc[len(data_one)] = zeros
        data_one = data_one.iloc[0:3000]
        temp_frame = []
        for i in range(5):
            temp = data_one.iloc[int(i * 600 + first_chopped_off): int((i + 1) * 600 - last_chopped_off)]
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
    percentage_removed_total = (int(18000 * 0.7) - len(index_to_be_removed)) / 18000.0
    print("percentage of data left for subject", subject_id, "is", percentage_removed_total)
    if percentage_removed_total < 0.35:
        print("the subject", subject_id, "should be removed and will be ignored")
        continue
    if str(subject_id) not in subject_unknown_percentage:
        subject_unknown_percentage[str(subject_id)] = {}
    subject_unknown_percentage[str(subject_id)]["known"] = percentage_removed_total
    subject_preprocess_record[str(subject_id)] = {}

    # checks each six session:
    print("check session for subject", subject_id)
    session_list = [i for i in range(0, 6)]
    for session in range(0, 6):
        session_lowerbound = 2100 * session
        session_upperbound = 2100 * (session + 1)
        to_be_removed = [i for i in index_to_be_removed if i >= session_lowerbound and i < session_upperbound]
        print("Number of rows to be removed for session", (session + 1), "is", len(to_be_removed))
        percentage_remained = (2100 - (len(to_be_removed))) / 3000.0
        print("percent of rows left in sesssion", (session + 1), "is", percentage_remained)
        if percentage_remained < 0.35:
            session_list.remove(session)
            print("session", session, " should be removed and will be ignored")
            print()
        print()
    if len(session_list) == 0:
        print("all sessions are ignored. Continue to next person")
        continue

    subject_preprocess_record[str(subject_id)]["session_remained"] = len(session_list)

    # cut 7 folds
    test1data = [];
    test2data = [];
    test3data = [];
    test4data = [];
    test5data = [];
    test6data = [];
    test7data = [];
    removed_dict = {}
    fold_names = ["fold1", "fold2", "fold3", "fold4", "fold5", "fold6", "fold7"]
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
        check_removed_index("fold1", removed_dict, index_to_be_removed, lowerBound, lowerBound + 60)
        test2data.append(data.iloc[lowerBound + 60: lowerBound + 120])
        check_removed_index("fold2", removed_dict, index_to_be_removed, lowerBound + 60, lowerBound + 120)
        test3data.append(data.iloc[lowerBound + 120: lowerBound + 180])
        check_removed_index("fold3", removed_dict, index_to_be_removed, lowerBound + 120, lowerBound + 180)
        test4data.append(data.iloc[lowerBound + 180: lowerBound + 240])
        check_removed_index("fold4", removed_dict, index_to_be_removed, lowerBound + 180, lowerBound + 240)
        test5data.append(data.iloc[lowerBound + 240: lowerBound + 300])
        check_removed_index("fold5", removed_dict, index_to_be_removed, lowerBound + 240, lowerBound + 300)
        test6data.append(data.iloc[lowerBound + 300: lowerBound + 360])
        check_removed_index("fold6", removed_dict, index_to_be_removed, lowerBound + 300, lowerBound + 360)
        test7data.append(data.iloc[lowerBound + 360: lowerBound + 420])
        check_removed_index("fold7", removed_dict, index_to_be_removed, lowerBound + 360, lowerBound + 420)

    folds_list = [test1data, test2data, test3data, test4data, test5data, test6data, test7data]
    # check folds percentages
    for name in fold_names:
        removed_num = removed_dict[name]
        remained_percentage = (((2100 * len(session_list)) / 7.0) - removed_num) / ((3000 * len(session_list)) / 7.0)
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
    print(removed_dict)
    print(sum(removed_dict.values()))
    for fold in folds_list:
        data = pd.concat(fold)
        print(data.shape)
        folds.append(data)
    if len(folds) == 0:
        print("all folds are ignored. Continue to next person")
        continue

    subject_preprocess_record[str(subject_id)]["folds_remained"] = len(folds)

    subject_prediction[str(subject_id)] = {}
    models = zip(names, classifiers, dicts_records)
    for name, classifier, dicts_record in models:
        accuracy = 0
        t0 = time()
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
            if name == "GradientBoosting":
                y_predict = clf.predict(X_test)
                # accuracy = accuracy + clf.score(X_test, y_test)
                accuracy += metrics.accuracy_score(y_test, y_predict)
            else:
                y_predict = clf.predict(X_test)
                if name == 'RandomForest':
                    Random_Forest_predicted_y.extend(y_predict)
                if name == 'RBF SVM':
                    RBF_SVM_predicted_y.extend(y_predict)
                # accuracy = accuracy + calculate_accuracy(y_test.tolist(), y_predict)
                accuracy += metrics.accuracy_score(y_test, y_predict)
        t1 = time()
        time_elapsed = t1 - t0
        print()
        print("The time it takes to run " + name + " is", time_elapsed)
        if name not in time_classifier:
            time_classifier[name] = 0
        time_classifier[name] = time_classifier[name] + time_elapsed
        accuracy = accuracy / float(len(folds_list))
        subject_prediction[str(subject_id)][name] = {}
        subject_prediction[str(subject_id)][name]["acutual_y"] = y_test
        subject_prediction[str(subject_id)][name]["predicted_y"] = y_predict
        dicts_record[str(subject_id)] = accuracy
        print("The accuracy of subject", subject_id, "is", accuracy, "with the model " + name)

print("Dicts_order is:")
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

dict_sum_recorder = dict(sorted(dict_sum_recorder.items(), key=lambda item: -item[1]))
print("The dic_sum_recorder is")
print(dict_sum_recorder)
print()
classifier_order = list(dict_sum_recorder.keys())
print("the order of the classifier is: ")
print(classifier_order)
print()
best_classifier_dict_sorted = dict(sorted(best_classifier_dict.items(), key=lambda item: -item[1]))
subject_id_order = list(best_classifier_dict_sorted.keys())  # The x axis of the plot
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
    ax.plot(x_axis, y, marker='D', label = label_name + "(" + str(temp_avg)+")")

ax.set_position([0.1,0.5, 1.2, 1.0])
ax.legend(loc='upper left')
plt.axhline(y=0.2, color='r', linestyle=':')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Subject ID orderd by ' + best_classifier_name, fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.savefig("output/"+data_source + "_" +data_status+"results/algorithm_comparison_each_subject.jpg", bbox_inches='tight', dpi = 2000)
plt.show()
