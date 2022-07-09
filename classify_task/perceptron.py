from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#################### データセット準備 ####################
datasets = load_iris()

x = datasets.data
y = datasets.target

x_train, x_test,y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1, stratify=y)

#################### 前処理 ####################
sc = StandardScaler() #標準化関数
sc.fit(x_train) #平均と標準偏差を計算

#データを標準化
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

#################### モデル作成 ####################
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(x_train_std, y_train)

#################### モデル性能評価 ####################
y_pred = ppn.predict(x_test_std)
score = accuracy_score(y_test,y_pred)
print('Accuracy: %.3f' % score)

