from PIL import Image
import numpy as np
from gapy import gago, bits2bytes
import os
import json

# ----------------------------
# 1. Funções auxiliares
# ----------------------------

def load_image(file_path):
    """Carrega imagem e transforma em array binário"""
    img = Image.open(file_path).convert('L')
    img_array = np.array(img)
    binary_array = np.where(img_array < 128, 1, 0)
    return binary_array

def loss_function(output, target):
    error = output - target
    return (error[0]**2 + error[1]**2 + 2*error[2]**2) / 4


# ----------------------------
# 3. Definir rede neural
# ----------------------------

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Inicializa a rede neural.
        input_size: número de neurônios na camada de entrada (tamanho da imagem flatten)
        hidden_sizes: lista com o número de neurônios em cada camada oculta
        output_size: número de neurônios na saída (x, y, r)
        """
        self.hidden_sizes = hidden_sizes
        self.weights = []
        self.biases = []

        prev_size = input_size
        for h in hidden_sizes:
            self.weights.append(np.random.randn(prev_size, h) * 0.01)
            self.biases.append(np.random.randn(h) * 0.01)
            prev_size = h

        self.weights.append(np.random.randn(prev_size, output_size) * 0.01)
        self.biases.append(np.random.randn(output_size) * 0.01)
    
    def sigmoid(self, x):
        """Função de ativação sigmóide"""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x):
        a = x
        for i in range(len(self.hidden_sizes)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
        # camada de saída linear
        output = np.dot(a, self.weights[-1]) + self.biases[-1]
        return output

    def get_weights(self):
        """Retorna todos os pesos e biases concatenados em vetor"""
        vec = []
        for w, b in zip(self.weights, self.biases):
            vec.append(w.flatten())
            vec.append(b)
        return np.concatenate(vec)

    def set_weights(self, vector):
        idx = 0
        for i in range(len(self.weights)):
            w_shape = self.weights[i].shape
            b_shape = self.biases[i].shape

            n_w = np.prod(w_shape)
            self.weights[i] = vector[idx:idx+n_w].reshape(w_shape)
            idx += n_w

            n_b = np.prod(b_shape)
            self.biases[i] = vector[idx:idx+n_b]
            idx += n_b


# ----------------------------
# 3. Configurações da rede
# ----------------------------

input_size = 24 * 24  # imagem 63x63 flatten
hidden_sizes = [5, 3]  # ex.: duas camadas ocultas, 5 e 3 neurônios
output_size = 3         # saída: x, y, r

nn = NeuralNetwork(input_size, hidden_sizes, output_size)


# ----------------------------
# 4. Ler metadados JSON
# ----------------------------

annotations_path = "amostras/dados/annotations.jsonl"
with open(annotations_path, 'r') as f:
    lines = f.readlines()

annotations = [json.loads(line) for line in lines]

# ----------------------------
# 5. Loop pelas imagens com aprendizado contínuo
# ----------------------------

# Inicializa população do GA como vazia
initial_population = []

for ann in annotations:
    file_path = os.path.join("amostras/dados", ann["file"])
    binary_array = load_image(file_path)

    # Extrair valores reais da primeira bola (n_bolas=1)
    circle = ann["circles"][0]
    x_real, y_real, r_real = circle["cx"], circle["cy"], circle["r"]

    # Normalização 0-1
    target = np.array([x_real/24, y_real/24, r_real/24])

    # Função fitness para GA
    def fit_func(bits):
        weight_vector = bits2bytes(bits, 'int16').astype(np.float32) / 1000.0
        nn.set_weights(weight_vector)
        output_pred = nn.forward(binary_array.flatten())
        # normaliza saída
        output_pred_norm = output_pred / 24.0
        loss = loss_function(output_pred_norm, target)
        return loss

    # Opções do GA
    gaoptions = {
        "PopulationSize": 200,
        "Generations": 50,
        "InitialPopulation": initial_population,  # usa população da iteração anterior
        "MutationFcn": 0.1,
        "EliteCount": 2,
    }

    num_weights = nn.get_weights().size
    num_bits = num_weights * 16

    # Executa GA
    result = gago(fit_func, num_bits, gaoptions)

    print(result)

    # Atualiza os melhores pesos na rede
    best_bits = result[0]
    best_weights = bits2bytes(best_bits, 'int16').astype(np.float32) / 1000.0
    nn.set_weights(best_weights)

    # Salva população final para próxima imagem (se gapy devolver população)
    if len(result) > 1:
        initial_population = result[0]

    # Testa a rede
    output_pred = nn.forward(binary_array.flatten())
    output_pred_denorm = output_pred
    print(f"Imagem: {file_path}")
    print(f"Valores reais: {(x_real, y_real, r_real)}")
    print(f"Output da rede (denormalizado): {output_pred_denorm}")
    print("--------")
