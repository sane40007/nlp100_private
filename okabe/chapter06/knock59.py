"""
59. ハイパーパラメータの探索
学習アルゴリズムや学習パラメータを変えながら，カテゴリ分類モデルを学習せよ．
検証データ上の正解率が最も高くなる学習アルゴリズム・パラメータを求めよ．
また，その学習アルゴリズム・パラメータを用いたときの評価データ上の正解率を求めよ．
"""

from cmath import log
from knock51 import X_train, X_valid, train, valid
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# 最適化関数
def objective(trial):

    # ハイパーパラメータのセット
    C = trial.suggest_float("C", 1e-4, 1e4, log=True)

    # モデルの学習
    svc = SVC(random_state=1, max_iter=10000, C=C, gamma="auto")
    svc.fit(X_train, train["CATEGORY"])

    # 正解率の算出
    valid_accuracy = svc.score(X_valid, valid["CATEGORY"])

    return valid_accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=4)

print("最適値：", study.best_value)
print("最適パラメータ", study.best_params)


"""
results

[I 2024-06-03 14:38:58,936] A new study created in memory with name: no-name-0d7e2502-c90e-46ad-82d1-904434d65508
[I 2024-06-03 14:42:48,470] Trial 0 finished with value: 0.42203898050974514 and parameters: {'C': 0.12128102271323626}. Best is trial 0 with value: 0.42203898050974514.
[I 2024-06-03 14:46:37,809] Trial 1 finished with value: 0.42203898050974514 and parameters: {'C': 0.2768293440267968}. Best is trial 0 with value: 0.42203898050974514.
[I 2024-06-03 14:51:50,832] Trial 2 finished with value: 0.42203898050974514 and parameters: {'C': 0.01653053751584299}. Best is trial 0 with value: 0.42203898050974514.
[I 2024-06-03 14:55:36,048] Trial 3 finished with value: 0.42203898050974514 and parameters: {'C': 0.4854935987503668}. Best is trial 0 with value: 0.42203898050974514.
最適値： 0.42203898050974514
最適パラメータ {'C': 0.12128102271323626}

"""