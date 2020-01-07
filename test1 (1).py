import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, accuracy_score,roc_auc_score,f1_score
import numpy as np
from sklearn.metrics import *
from sklearn import metrics
data = pd.read_excel("8.19str_to_number.xls")
# data = data["直接胆红素", "甘油三酯", "抑酶", "总胆汁酸", "抑酶药物", "APTT",
#              "LDH", "年龄", "尿素",
#              "是否有术后胰腺炎"]
names=['ALB', '胰管造影剂注入', '胃切除史', '性别', '胆管支架', '困难插管', 'TC', '饮酒史']

x_train, x_test, y_train, y_test = train_test_split(data[names], data["是否有术后胰腺炎"], random_state=0)

data_train = xgboost.DMatrix(x_train, y_train)  # 使用XGBoost的原生版本需要对数据进行转化
data_test = xgboost.DMatrix(x_test, y_test)

# param = {'max_depth': 4, 'eta': 0.5, 'objective': 'binary:logistic'}
param={'max_depth':3}
watchlist = [(data_test, 'test'), (data_train, 'train')]
n_round = 20
booster = xgboost.train(param, data_train, num_boost_round=n_round, evals=watchlist)

y_pred = booster.predict(data_test)
y = data_test.get_label()
print(y_pred)
accuracy = sum(y == (y_pred > 0.5))
accuracy_rate = float(accuracy) / len(y_pred)
print('样本总数：{0}'.format(len(y_pred)))
print('正确数目：{0}'.format(accuracy))
print('正确率：{0:.3f}'.format((accuracy_rate)))
print("概率大于0.5的", y_pred[y_pred > 0.5])

y_test_true = y_test.values.tolist()
y_pred_result = [1 if x > 0.5 else 0 for x in y_pred.tolist()]

target_names = ['class 0', 'class 1']
print(classification_report(y_test_true, y_pred_result, target_names=target_names))
print('AUC: %.4f' % roc_auc_score(y_test, y_pred))
Accuracy = accuracy_score(np.array(y_test), y_pred_result )  # 计算准确度
Accuracy = round(Accuracy, 3)
Precision = precision_score(y_test, y_pred_result )  # 计算精确度
Precision = round(Precision, 3)
Recall = recall_score(np.array(y_test), y_pred_result )  # 计算召回率
Recall = round(Recall, 3)
F1 = f1_score(np.array(y_test), y_pred_result )
F1 = round(F1, 3)
matr=metrics.confusion_matrix(y_test, y_pred_result , labels=None, sample_weight=None)
print(Accuracy, Precision, Recall, F1)
print(matr)