# Sentiment Analysis Building Model

import numpy as np
import pandas as pd
from collections import Counter
import helper
import pickle

# ----------------------------------------------------
print('Leitura de dados')
data = pd.read_csv('data/imdb_labelled.txt', sep='\t')

print('Pré-processamento')
data['Review_preprocessado'] = data.Review.apply(helper.preprocessar_sentenca)

print('Construção do Modelo')
counter_positivos = Counter()
counter_negativos = Counter()
counter_todos = Counter()

# Implemente sua resposta aqui
for i in range(len(data.Review_preprocessado)):
    if(data.Label[i] == 0):
        for word in data.Review_preprocessado[i]:
            counter_negativos[word] += 1
            counter_todos[word] += 1
    else:
        for word in data.Review_preprocessado[i]:
            counter_positivos[word] += 1
            counter_todos[word] += 1    


taxa_pos_neg = {}

# Filtrar termos que apareceram pelo menos 10 vezes
# e calcular a taxa de positivo negativo
valor_minimo = 10
for word, contagem in list(counter_todos.most_common()):
    if(contagem >= valor_minimo):
        taxa_pos_neg[word] = round( counter_positivos[word] / float(counter_negativos[word]+1), 3 )


score_pos_neg = {}

# Transformar taxa positivo negativo usando log
for word, taxa in taxa_pos_neg.items():    
    if (taxa > 1):
        score_pos_neg[word] = round( np.log(taxa), 3)
    elif (taxa < 1):
        score_pos_neg[word] = round( -np.log(1/(taxa + 0.001)), 3)
    else:
        score_pos_neg[word] = 0

# Nosso modelo é o score_pos_neg

model = score_pos_neg
model_filename = 'model.pkl'
print('Salvando o modelo: ' + model_filename)
model_file = open(model_filename, 'wb')
pickle.dump(model, model_file)