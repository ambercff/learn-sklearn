import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1 - Carregamento de Dados
# Utiliza-se o dataset Titanic
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# 2 - Printando os dados para inspecionar
print(data.head())

# 3 - Pré-processamento de dados
# Seleciona-se o que é relevante para o algoritmo
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
target = 'Survived'

# Tratando os valores ausentes, se caso não tiver idade o valor é substituído por uma média e se não tiver o porto de embarque o valor é substituído pelo mais frequente
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna('S')
data['Cabin'] = data['Cabin'].fillna('Unknown')

# Transformando as variáveis categóricas para números
label_enc = LabelEncoder()
data['Sex'] = label_enc.fit_transform(data['Sex'])
data['Embarked'] = label_enc.fit_transform(data['Embarked'])
data['Fare'] = data['Fare'].fillna(data['Fare'].median())

# Separando os dados em x (características) e y (alvo)
x = data[features]
y = data[target]

# Escalar os dados numéricos para normalizar idade e tarifa, assim as diferenças de escala são evitadas e não prejudicam o modelo
scaler = StandardScaler()
# Utiliza-se o .loc para selecionar as colunas de forma explícita, deixando claro que você deseja modificar os dados originais
x.loc[:, ['Age', 'Fare']] = scaler.fit_transform(x[['Age', 'Fare']])

# 4 - Dividindo os dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# 5 - Treinando o modelo
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# 6 - Fazendo as previsões
y_pred = model.predict(x_test)

# 7 - Avaliando o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy: .2f}")

# Exemplo de predição com novos dados
new_passenger = pd.DataFrame({
    'Pclass': [3],
    'Sex': [1],
    'Age': [25],
    'Fare': [7.25],
    'Embarked': [2]
})

new_passenger[['Age', 'Fare']] = scaler.transform(new_passenger[['Age', 'Fare']])
prediction = model.predict(new_passenger)
print(f"Predição para o novo passageiro: {'Sobreviveu' if prediction[0] == 1 else 'Não sobreviveu'}")