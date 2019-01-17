import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, f1_score, recall_score, precision_score, confusion_matrix
from pandas_ml import ConfusionMatrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import cycle



# load data
data = pd.read_csv(r"C:\Users\User\PycharmProjects\pointnet\result_df.csv")
data.columns

original_label = data['True_labels']
original_label = original_label.values.astype(int)
len(original_label)

test_label = data['Pred_labels']
test_label = test_label.values.astype(int)


# precision, recall, F1-score
print(metrics.classification_report(original_label,test_label))

# confusion matrix
cm = ConfusionMatrix(np.array(original_label), test_label)
print(cm)


# precison-recall curve
score = np.load(r"C:\Users\User\PycharmProjects\pointnet\result_softmax.npy")


# make label
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

onehot_y = encode_onehot(original_label)



# precision recall curve
precision = dict()
recall = dict()
average_precision = dict()
for i in range(4):
    precision[i], recall[i], _ = precision_recall_curve(onehot_y[:, i],
                                                        score[:, i])
    average_precision[i] = average_precision_score(onehot_y[:, i], score[:, i])



# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(onehot_y.ravel(),score.ravel())
average_precision["micro"] = average_precision_score(onehot_y, score,average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

plt.figure(figsize=(7, 7))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []

l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'.format(average_precision["micro"]))


linestyles = cycle(['-', '--', '-.', ':'])
colors = ['darkorange', 'cornflowerblue', 'purple', 'green']
colors = ['darkorange', 'cornflowerblue', 'purple', 'green']
classes = ['DoubleDoor','RevolvingDoor', 'SingleDoor', 'SlidingDoor']

for i, line in zip(range(4), linestyles):
    l, = plt.plot(recall[i], precision[i], color=colors[i], linestyle=line, lw=2)
    lines.append(l)
    labels.append('{0} (area = {1:0.2f})'
                  ''.format(classes[i], average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PointNet Precision-Recall curve')
plt.legend(lines, labels, loc=(0.05, 0.05), prop=dict(size=12))
plt.show()




# Compute ROC curve and ROC area for each class
n_classes = 4
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(onehot_y[:, i], score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(7, 7))

classes = ['DoubleDoor','RevolvingDoor', 'SingleDoor', 'SlidingDoor']
linestyles = cycle(['-', '--', '-.', ':'])
colors = ['darkorange', 'cornflowerblue', 'purple', 'green']
for i, line in zip(range(n_classes), linestyles):
    plt.plot(fpr[i], tpr[i], color=colors[i], linestyle=line,
             label=' {0} (area = {1:0.2f})'
                   ''.format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('PointNet ROC curve')
plt.legend(loc="lower right")
plt.show()
