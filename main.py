import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor


# ----------------------------------------------------------------------
# Литвинов Илья. Тестовое задание №3.
#
# Цель задачи состоит в определении принадлежности произвольного игрока
#  к заранее обозначенной группе. Я решил подойти к решению как к
#  задаче поиска аномалий, а конкретнее - поиска новизны.
# ----------------------------------------------------------------------

# В этом блоке данные из таблиц разделяются.
# all_dataset будет использоваться для визуализации. В работе самого алгоритма не участвует.
# prediction_data это данные игроков, которых мы хотим проверить на принадлежность,
# в данном случае, это зелёные игроки из таблицы с заданием.
# training_data это данные игроков, которые образуют группу. Их показатели определяют
# эталон.
data = pd.read_excel('data.xlsx', sheet_name='Group_Style_By_stats')
data = data.dropna(axis=1)
all_dataset = data.values[:, 1:]

new_player = data.iloc[49:53]
prediction_data = new_player.values[:, 1:]

old_player = data.drop(labels=[49, 50, 51, 52])
training_data = old_player.values[:, 1:]

# Для определения принадлежности я использовал LocalOutlierFactor.
# Его идея основана на оценке изолированности объекта по отношению к окружению,
# таким образом, можно определять выбросы и новизну.
# Для обучения в алгоритм подаются данные старых игроков. Далее, для новых игроков
# мы определяем, соответствуют ли они группе, на которой обучался алгоритм.
# Параметр novelty=True, задаёт работу алгоритма как определение новизны,
# все данные, которые соответствуют группе возвращают 1, а неподходящие -1.
# В качестве альтернативы, я рассматривал IsolationForest, результаты работы
# алгоритмов совпали.
clf = LocalOutlierFactor(n_neighbors=2, novelty=True).fit(training_data)
labels = clf.predict(prediction_data)


def distribution(row, column):
    """Интерпретация лейблов"""
    if row[column] == -1:
        return 'No'
    if row[column] == 1:
        return 'Yes'


results = pd.DataFrame()

# Компановка фрейма с результатами игроков.
results['player'] = new_player['player']
results['groupNum'] = labels
results['Result'] = results.apply(distribution, axis=1, column='groupNum')
print(results)

# Блок с рисованием графика, я использовал его как вспомогательное средство,
# по этому покраска игроков выполнена для конкретной задачи.
embedded = TSNE(n_components=2, learning_rate='auto', init='random', random_state=2).fit_transform(all_dataset)

data['X'] = embedded[:, 0]
data['Y'] = embedded[:, 1]


def colors(row, column):
    """Покраска точек на графике в соответствии с именем"""
    if row[column] == 'new_player1':
        return 'Red'
    if row[column] == 'new_player2':
        return 'Green'
    if row[column] == 'new_player3':
        return 'Brown'
    if row[column] == 'new_player4':
        return 'Black'
    else:
        return 'Blue'


# Покраска точек и рисование графика
data["Color"] = data.apply(colors, axis=1, column='player')

plt.scatter(data['X'], data['Y'], s=10, c=list(data['Color']))
plt.show()
