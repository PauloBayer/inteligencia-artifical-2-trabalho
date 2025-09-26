from gapy import gago, bits2bytes

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
    """
    Função de adaptação (fitness function) para Algoritmo Genético.
    
    Objetivo
    --------
    Definir valores inteiros para que a seguinte equação seja verdadeira:
        
    .. math:: x_1 + x_2 = |x_3 - x_4|
        
    Entrada
    -------
    64 bits (8 bytes). Cada 16 bits (2 bytes) formam um valor inteiro.
    
    Saída
    -----
    Erro absoluto em relação ao resultado esperado.
    """
    # Converte a sequência de bits em números inteiros de 16 bits
    X = bits2bytes(bits, 'int16').astype(int)
    # Calcula o erro com base nos números inteiros
    
    erro = abs((X[0] + X[1]) - abs(X[2] - X[3]))
    return erro

# Executa o algoritmo genético com a função de adaptação, 64 bits por indivíduo
# e as opções definidas
result = gago(fit_func, 64, gaoptions)

# Extrai os bits do melhor indivíduo encontrado pelo algoritmo genético
bits = result[0]

# Converte os bits extraídos em números inteiros de 16 bits
X = bits2bytes(bits, 'int16').astype(int)

# Imprime o resultado (os números inteiros encontrados)
print('Resultado =', list(X))

# Calcula e imprime o erro para o melhor indivíduo encontrado
print('Erro =', fit_func(bits))

# Imprime a soma dos dois primeiros números inteiros encontrados
print('x1 + x2 =', X[0] + X[1])

# Imprime o valor absoluto da diferença entre os dois últimos números inteiros
# encontrados
print('|x3 - x4| =', abs(X[2] - X[3]))

print('X1: ', X[0])
print('X2: ', X[1])
print('X3: ', X[2])
print('X4: ', X[3])
