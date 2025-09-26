from gapy import gago, bits2bytes
import warnings
import numpy as np

# Opções para o algoritmo genético
gaoptions = {
    "PopulationSize": 200,    # Tamanho da população
    "Generations": 50,        # Número de gerações
    "InitialPopulation": [],  # População inicial (vazia neste caso)
    "MutationFcn": 0.15,      # Taxa de mutação
    "EliteCount": 2,          # Número de indivíduos elite a serem mantidos
}

# Função de adaptação (fitness function)
def fit_func(bits):
    X = bits2bytes(bits, 'float16').astype(float)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        erro = abs(np.subtract(np.add(X[0], X[1]), np.abs(np.subtract(X[2], X[3]))))
    return erro

# Executa o algoritmo genético com a função de adaptação, 64 bits por indivíduo
# e as opções definidas
result = gago(fit_func, 64, gaoptions)

# Extrai os bits do melhor indivíduo encontrado pelo algoritmo genético
bits = result[0]

# Converte os bits extraídos em números inteiros de 16 bits
X = bits2bytes(bits, 'float16').astype(float)

# Imprime o resultado (os números inteiros encontrados)
print('Resultado =', list(X))

# Calcula e imprime o erro para o melhor indivíduo encontrado
print('Erro =', fit_func(bits))

# Imprime a soma dos dois primeiros números inteiros encontrados
print('x1 + x2 =', X[0] + X[1])

# Imprime o valor absoluto da diferença entre os dois últimos números inteiros
# encontrados
print('|x3 - x4| =', abs(X[2] - X[3]))
