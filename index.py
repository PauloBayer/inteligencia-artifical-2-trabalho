from PIL import Image
import numpy as np

# Fluxo geral:
# 1. Coletar imagens
# 2. Pré processamento (redimensionar para ficar mais leve)
# 3. Definir rede neural
# 4. Definir função de perda
# 5. Implementar algotritmo de genético (vai ser usado para otimizar os pesos da rede neural)
# 6. Treinar a rede neural

# ---------- 1. Coletar imagens ----------

file_path = "amostras/dados/images/single/single_0000.png"
img = Image.open(file_path).convert('L')

img_array = np.array(img)
binary_array = np.where(img_array < 128, 1, 0)

print(binary_array.shape)
print(binary_array)

# ---------- 2. Pré processamento (redimensionar para ficar mais leve) ----------

# ---------- 3. Definir rede neural ----------

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Inicializa a rede neural.
        input_size: número de neurônios na camada de entrada (tamanho do bloco flatten)
        hidden_size: número de neurônios na camada oculta
        output_size: número de neurônios na saída (1 = presença de bola)
        """
        # Pesos aleatórios para Entrada -> Oculta
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        
        # Pesos aleatórios para Oculta -> Saída
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.random.randn(output_size)
    
    def sigmoid(self, x):
        """Função de ativação sigmóide"""
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x):
        """
        Propagação da entrada pela rede
        x: vetor de entrada (ex: 64 pixels do bloco)
        retorna: valor de saída (0 a 1)
        """
        # Camada oculta
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Camada de saída
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2

    def get_weights(self):
        """Retorna todos os pesos e biases como um vetor"""
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])
    
    def set_weights(self, weight_vector):
        """Atualiza os pesos e biases a partir de um vetor"""
        idx = 0
        
        # W1
        n_W1 = self.W1.size
        self.W1 = weight_vector[idx:idx+n_W1].reshape(self.W1.shape)
        idx += n_W1
        
        # b1
        n_b1 = self.b1.size
        self.b1 = weight_vector[idx:idx+n_b1]
        idx += n_b1
        
        # W2
        n_W2 = self.W2.size
        self.W2 = weight_vector[idx:idx+n_W2].reshape(self.W2.shape)
        idx += n_W2
        
        # b2
        n_b2 = self.b2.size
        self.b2 = weight_vector[idx:idx+n_b2]

input_size = 255 * 255   # tamanho do bloco 8x8
hidden_size = 10  # camada oculta com 10 neurônios
output_size = 1   # saída: 0 ou 1 (bola ou não)

nn = NeuralNetwork(input_size, hidden_size, output_size)

print("Pesos iniciais:", nn.get_weights())

output = nn.forward(binary_array.flatten())
print("Output da rede:", output)

# ---------- 4. Definir função de perda ----------

def loss_function(nn, X, Y):
    return 1

# ---------- 5. Implementar algotritmo de genético (vai ser usado para otimizar os pesos da rede neural) ----------

# ---------- 6. Treinar a rede neural ----------
