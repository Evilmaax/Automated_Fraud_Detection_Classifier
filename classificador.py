import os, random, sys, time
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV as CV
from sklearn.metrics import plot_confusion_matrix, recall_score, matthews_corrcoef

raiz_projeto = os.path.dirname(sys.argv[0])

try:
    file = open(raiz_projeto + "/config.txt", "r")
    config = file.readlines()

    rotulo = config[0].split(" #")[0]
    to_keep = config[1].split(" #")[0].split(',')
    tx_correlacao = float(config[2].split(" #")[0])
    eta = float(config[3].split(" #")[0])
    n_estimators = int(config[4].split(" #")[0])
    min_child_weight = int(config[5].split(" #")[0])
    subsample = float(config[6].split(" #")[0])
    colsample_bytree = float(config[7].split(" #")[0])
    scale_pos_weight = int(config[8].split(" #")[0])
    max_depth = int(config[9].split(" #")[0])
    split = float(config[10].split(" #")[0])

    file.close()

except:
    print("\nNão foi possível carregar o arquivo de configurações\nPor favor verifique")
    sys.exit()


def carregar():
    try:
        arquivo = os.listdir(f"{raiz_projeto}/Dataset")

        print("\nQual arquivo você quer usar nesta operação?\n")
        for i in range(len(arquivo)):
            print(f"{i + 1} - {arquivo[i]}")
        op = int(input("\nResposta: "))

        print("\nCarregando dados.\nIsto pode demorar de acordo com o tamanho do arquivo")
        data = pd.read_csv(raiz_projeto + '/Dataset/' + arquivo[op - 1])
        print("\nArquivo " + arquivo[op - 1] + " carregado com sucesso")

        return data

    except:
        print("\nNão foi possível carregar o arquivo\nPor favor verifique se o mesmo encontra-se na"
              " pasta 'Dataset' dentro da raiz do projeto e se possui a extensão .CSV")
        sys.exit()


def otimizar():
    data = carregar()

    cv = int(input("\nQuantos folds por rodada?"
                   "\nMelhora significativa nos resultados a partir de 10 folds: "))

    round, tempoTotal = 1, time.time()

    parametros = [['eta', 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                  ['n_estimators', 300, 400, 500, 600, 700, 800],
                  ['min_child_weight', 3, 4, 5, 6, 7],
                  ['subsample', 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  ['colsample_bytree', 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                  ['scale_pos_weight', 3, 4, 5, 6, 7],
                  ['max_depth', 5, 6, 7, 8, 9, 10],
                  ['split', 0.2, 0.25, 0.3, 0.35]]

    disponivel = []

    for x in range(len(parametros)):
        if len(parametros[x]) > 2:
            disponivel.append(x)

    while len(disponivel) > 0:
        print("\nRound ", round)
        start = time.time()
        nomes, param = [], []

        if len(disponivel) > 3:
            temp = random.sample(disponivel, 3)  # escolhe 3 parametross random
        else:
            temp = random.sample(disponivel, len(disponivel))  # ou quantos parametross ainda tiverem

        for x in temp:
            nomes.append(parametros[x][0])
        for x in range(len(temp)):  # Popular 3 listas
            listaTemp = []
            if len(parametros[temp[x]]) > 3:
                temp2 = random.sample(range(1, len(parametros[temp[x]])), 3)  # escolhe 3 valores random
            else:
                temp2 = random.sample(range(1, len(parametros[temp[x]])), len(parametros[temp[x]]) - 1)

            for y in range(len(temp2)):
                listaTemp.append(parametros[temp[x]][temp2[y]])
            param.append(listaTemp)

        param = dict(zip(nomes, param))

        print("    Valores testados: ", param)
        grid = CV(estimator=XGBClassifier(verbosity=0, use_label_encoder=False),
                  param_grid=param,
                  scoring='recall',
                  n_jobs=-1,
                  cv=cv)

        grid.fit(data.drop([rotulo], axis='columns'), data[rotulo])
        melhores = grid.best_params_
        print("    Melhores parâmetros: ", melhores)

        for chave, valor in melhores.items():
            for x in range(len(parametros)):
                if parametros[x][0] == chave:
                    y = 0
                    while y < len(parametros[x]):
                        if (parametros[x][y] != valor) and (parametros[x][y] in param[chave]):
                            parametros[x].pop(y)
                            y = 0
                        y += 1

        round += 1
        print(f"    Tempo necessário para o round de treinamento: {time.time() - start:.3f} segundos")

        disponivel = []

        for x in range(len(parametros)):
            if len(parametros[x]) > 2:
                disponivel.append(x)

    print(f"\nTempo total: {time.time() - tempoTotal:.3f} segundos")

    with open("otimizado.txt", "w") as novaConfig:
        for linha in parametros:
            linha = " ".join(map(str, linha))
            linha1, linha2 = linha.split(' ')
            linha2 = linha2.strip() + " # " + linha1 + "\n"
            novaConfig.write(linha2)

    file = open(raiz_projeto + "/otimizado.txt", "r")
    configOtimizada = file.readlines()
    print("Arquivo otimizado.txt exportado para a pasta raiz da aplicação")


    eta = float(configOtimizada[0].split(" #")[0])
    n_estimators = int(configOtimizada[1].split(" #")[0])
    min_child_weight = int(configOtimizada[2].split(" #")[0])
    subsample = float(configOtimizada[3].split(" #")[0])
    colsample_bytree = float(configOtimizada[4].split(" #")[0])
    scale_pos_weight = int(configOtimizada[5].split(" #")[0])
    max_depth = int(configOtimizada[6].split(" #")[0])
    split = float(configOtimizada[7].split(" #")[0])

    print("\nOtimização concluída. Valores selecionados:\n\n"
          f"Eta: {eta}\n"
          f"n_estimators: {n_estimators}\n"
          f"min_child_weight: {min_child_weight}\n"
          f"subsample: {subsample}\n"
          f"colsample_bytree: {colsample_bytree}\n"
          f"scale_pos_weight: {scale_pos_weight}\n"
          f"max_depth: {max_depth}\n"
          f"taxa de split: {split}")

    file.close()

    input("\nPressione enter para continuar")
    menu()


def padronizar(data):

    file = open(raiz_projeto + "/colunas.txt", "r")
    colunas = file.read().splitlines()
    file.close()

    for i in data.columns:
        if i not in colunas:
            data.drop([i], axis='columns', inplace=True)
        else:
            data[i].fillna(0, inplace=True)

    return data


def classificar(modelo):
    tempoTotal = time.time()

    data = carregar()

    print("\nProcesso de classificação iniciado")
    data = padronizar(data)

    data[rotulo] = modelo.predict(data)
    data.to_csv(raiz_projeto + '/Dataset/resultado_classificacao.csv', index=False)

    print('\nArquivo resultado_classificacao.csv exportado para a pasta Dataset')
    print(f"Tempo total para a classificação: {time.time() - tempoTotal:.3f} segundos")


def treinar():
    tempoTotal = time.time()

    data = carregar()

    verdadeiro = data[rotulo].value_counts()[0]
    fraude = data[rotulo].value_counts()[1]
    total = data[rotulo].count()

    print(
        f'\nAlgumas informações sobre o seu dataset:\n\nTransações totais: {total}\nTransações verdadeiras: {verdadeiro}\nTransações fraudulentas: {fraude}\nPorcentagem de fraudes: {(fraude / total) * 100:.3f}%')

    bbc = BalancedBaggingClassifier(base_estimator=XGBClassifier(
        eta=eta,
        n_estimators=n_estimators,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        max_depth=max_depth,
        objective='binary:logistic',
        use_label_encoder=False,
        verbosity=0),
        sampling_strategy='all',
        n_jobs=-1)

    X_train, X_test, y_train, y_test = train_test_split(data.drop([rotulo], axis='columns'),
                                                        data[rotulo], test_size=split, stratify=data[rotulo])

    bbc.fit(X_train, y_train)
    joblib.dump(bbc, raiz_projeto + '/modelo.pkl')
    print('\nModelo gerado e exportado para a pasta raiz da aplicação')

    y_pred = bbc.predict(X_test)

    print("\nSeus valores de treinamento:")
    print("Mathews: ", matthews_corrcoef(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred))

    plot_confusion_matrix(bbc, X_test, y_test, values_format='d',
                          display_labels=['verdadeiras', 'fraudulentas'])

    print(f"\nTempo total: {time.time() - tempoTotal:.3f} segundos")

    plt.show()
    input("\nPressione enter para continuar")
    menu()


def preProcessamento():

    data = carregar()
    print("\nPré processamento dos dados iniciado")

    tempoTotal = time.time()
    correlacionadas = set()
    matriz_corr = data.corr()

    for i in range(len(matriz_corr.columns)):
        for j in range(i):
            if abs(matriz_corr.iloc[i, j]) > tx_correlacao:
                colname = matriz_corr.columns[i]
                if colname not in to_keep:
                    correlacionadas.add(colname)

    dados_sem_correlacoes = data.drop(correlacionadas, axis=1)
    data = dados_sem_correlacoes

    print("\nColunas correlacionadas removidas")
    nao_numerico = set()

    for y in data.columns:
        if data[y].dtype == object:
            nao_numerico.add(y)

    dados_sem_str = data.drop(nao_numerico, axis=1)
    data = dados_sem_str

    print("Colunas não numéricas removidas")

    for i in data.columns:
        data[i].fillna(0, inplace=True)

    print("Valores not a number preenchidos")

    with open("colunas.txt", "w+") as colunas:
        for i in data.columns:
            if i != rotulo:
                colunas.write(i)
                colunas.write("\n")

    data.to_csv(raiz_projeto + '/Dataset/pre_processamento_concluido.csv', index=False)
    print("\nPré processamento concluído")
    print("Arquivo pre_processamento_concluido.csv exportado para a pasta Dataset")
    print(f"\nTempo total: {time.time() - tempoTotal:.3f} segundos")

    input("\nPressione enter para continuar")
    menu()


def menu():
    opcoes, op = [1, 2, 3, 4, 5, 6, 7], 0

    while op not in opcoes:
        op = int(input("\nSelecione uma opção:\n\n"
                       "1 - Aplicar o pré processamento ao dataset (Escolha esta opção se for a primeira vez que utiliza o mesmo)\n"
                       "2 - Treinar o modelo utilizando a configuração padrão\n"
                       "3 - Treinar o modelo Utilizando a configuração otimizada\n"
                       "4 - Testar combinações de parâmetros e valores e gerar configuração otimizada\n"
                       "5 - Classificar transações usando o modelo treinado\n"
                       "6 - Ajuda\n"
                       "7 - Sobre esta aplicação\n\n"
                       "Resposta: "))

    if op == 1:
        preProcessamento()

    elif op == 2:
        try:
            file = open(raiz_projeto + "/config.txt", "r")
            config = file.readlines()

            eta = float(config[3].split(" #")[0])
            n_estimators = int(config[4].split(" #")[0])
            min_child_weight = int(config[5].split(" #")[0])
            subsample = float(config[6].split(" #")[0])
            colsample_bytree = float(config[7].split(" #")[0])
            scale_pos_weight = int(config[8].split(" #")[0])
            max_depth = int(config[9].split(" #")[0])
            split = float(config[10].split(" #")[0])

            file.close()

            print(f"\nValores do arquivo de configuração padrão carregados com sucesso:\n\n"
                  f"Eta: {eta}\n"
                  f"n_estimators: {n_estimators}\n"
                  f"min_child_weight: {min_child_weight}\n"
                  f"subsample: {subsample}\n"
                  f"colsample_bytree: {colsample_bytree}\n"
                  f"scale_pos_weight: {scale_pos_weight}\n"
                  f"max_depth: {max_depth}\n"
                  f"taxa de split: {split}")

        except:
            print("\nArquivo de configuração padrão não encontrado.\nVerifique a pasta raiz")

        treinar()

    elif op == 3:
        try:
            otimizado = open(raiz_projeto + "/otimizado.txt", "r")
            configPersonalizada = otimizado.readlines()
            print("\nArquivo de configuração otimizado carregado com sucesso")

            eta = float(configPersonalizada[0].split(" #")[0])
            n_estimators = int(configPersonalizada[1].split(" #")[0])
            min_child_weight = int(configPersonalizada[2].split(" #")[0])
            subsample = float(configPersonalizada[3].split(" #")[0])
            colsample_bytree = float(configPersonalizada[4].split(" #")[0])
            scale_pos_weight = int(configPersonalizada[5].split(" #")[0])
            max_depth = int(configPersonalizada[6].split(" #")[0])
            split = float(configPersonalizada[7].split(" #")[0])

            print(f"Valores do arquivo de configuração otimizado:\n\n"
                  f"Eta: {eta}\n"
                  f"n_estimators: {n_estimators}\n"
                  f"min_child_weight: {min_child_weight}\n"
                  f"subsample: {subsample}\n"
                  f"colsample_bytree: {colsample_bytree}\n"
                  f"scale_pos_weight: {scale_pos_weight}\n"
                  f"max_depth: {max_depth}\n"
                  f"taxa de split: {split}")

            otimizado.close()
            treinar()

        except:
            print("\nArquivo de configuração otimizada não encontrado.\n"
                  "Efetue a otimização através do menu")
            input("\nPressione enter para continuar")
            menu()

    elif op == 4:
        otimizar()

    elif op == 5:
        try:
            modelo = joblib.load("modelo.pkl")
            print("\nmodelo carregado com sucesso")

            classificar(modelo)

        except:
            print("\nModelo ainda não treinado.\n"
                  "Treine o modelo a partir do menu primeiramente")
            input("\nPressione enter para continuar")
            menu()

    elif op == 6:

        print('\nEsta ferramenta utiliza o algoritmo Extreme Gradient Boosting (XGB) em sua construção.\n'
               'Por ser um algoritmo de boosting, o XGB implementa diversos algoritmos mais simples com o intuito de chegar a um\n'
               'resultado mais completo e acurado no final.\n\n'
               'Na prática os algoritmos deste tipo funcionam como árvores de decisão sequenciais já que o valor que foi\n'
               'predito em n irá ser levado em conta para a predição em n+1 onde o algoritmo irá utilizar novas colunas e valores aleatórios\n'
               'para aprender a lidar com as particularidades das classificações ERRADAS da rodada anterior.\n\n'
               'Também foi utilizado neste trabalho a biblioteca Balanced Bagging Classification, para lidar com dados desbalanceados e\n'
               'e o Grid Search CV para fazer o cruzamento dos parâmetros e valores na opção 4 do menu, quando da busca dos valores otimizados\n'
               'Um dos pontos altos do XGB é justamente o fato dele possuir dezenas de parâmetros configuráveis. \n'
               'Estes são os principais parãmetros utilizados por esta ferramenta:\n\n'
               'Eta: Representa a taxa de aprendizagem, chamada de eta na documentação oficial do XGB.\n'
               'N_estimators: Refere-se ao número de árvores de decisão que serão criadas pelo modelo durante o treinamento.\n'
               'Min_child_weight: Define a soma mínima dos pesos necessária para a árvore seguir sendo particionada.\n'
               'Quanto mais alto este valor, mais conservador será o algoritmo e menos relações superespecíficas serão aprendidas.\n'
               'Max_depth: Representa a profundidade máxima da árvore. Assim como o parâmetro anterior, possui a mesma relação de controle com overfitting. '
               'Porém, neste caso, quanto mais alto mais tendência ao overfitting.\n'
               'Subsample: Determina a fração de dados de treinamento aleatória que será repassada a cada árvore antes delas aumentarem mais 1 nível.\n'
               'Colsample_bytree: Similar ao anterior, determina a fração de colunas que serão fornecidas aleatoriamente no momento de criação das árvores.\n'
               'Scale_pos_weight: Controla o balanço entre pesos positivos e negativos. É recomendado para casos com grande desbalanceamento entre classes.\n\n'
               '****************************************************\n'
               '                     IMPORTANTE\n\n'
              'Preencha o arquivo de configurações na pasta raiz da aplicação\n'
              'para o correto funcionamento da aplicação\n\n'
              'Para mais informações:\n'
              'Contato -> MaximilianoMeyer48@gmail.com\n'
              'Para atualizações, clonar ou contribuir com o projeto -> https://github.com/Evilmaax\n\n'               
              '****************************************************\n')

        x = input('Pressione enter para voltar ao menu')
        menu()


    elif op == 7:
        x = input('\nEsta aplicação foi desenvolvida por Maximiliano Meyer durante o trabalho de conclusão\n'
                  'intitulado "Desenvolvimento de uma ferramenta de identificação\n'
                  'de fraudes em trnasações com cartões de crédito com uso de Machine Learning"\n'
                  'desenvolvido no curso de Ciências da Computação da Universidade de Santa Cruz do Sul - Unisc\n'
                  'Pesquisa e projeto desenvolvidos e defendidos no primeiro semestre de 2021.\n\n'
                  'Seu uso é livre desde que informada a fonte.\n'
                  'Para atualizações, clonar ou contribuir com o projeto -> https://github.com/Evilmaax\n'
                  'Contato -> MaximilianoMeyer48@gmail.com\n\n'
                  'Versão 0.1\n\n'
                  'Pressione enter para voltar ao menu')

        menu()


try:
    modelo = joblib.load("modelo.pkl")
    op = 0

    while op != 1 and op != 2:
        op = int(input("\nVocê já possui um modelo treinado e pronto para uso.\n"
                       "Gostaria de:\n\n"
                       "1 - Classificar transações usando o modelo treinado anteriormente\n"
                       "2 - Preparar os arquivos e configurações de teste\n\n"
                       "Resposta: "))

    if op == 1:
        classificar(modelo)

    elif op == 2:
        menu()

except:
    menu()
