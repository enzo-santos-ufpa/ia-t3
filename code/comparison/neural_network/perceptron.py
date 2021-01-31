import math
import random
from typing import Optional, List

from comparison.neural_network.gradient import Gradient
from comparison.neural_network.utils import Vector, sigmoid


# noinspection NonAsciiCharacters,PyPep8Naming
class Perceptron:
    """
    Representa um perceptron (ou neurônio) de uma rede neural.

    Esse perceptron possui como função de ativação a função sigmoide.
    """

    __slots__ = ('bias', 'weights', 'last_input', 'last_output')

    def __init__(self, input_size: int, bias: float, *, weights: Optional[Vector] = None):
        self.bias: float = bias
        """
        Representa um bias a ser adicionado aos pesos em cada predição desse perceptron.
        """

        self.weights: Vector
        """
        Armazena os pesos a serem considerados a cada entrada fornecida a esse perceptron.
        """
        if weights is None:
            # He initialization
            c: float = math.sqrt(2.0 / input_size)
            self.weights = [random.normalvariate(mu=0, sigma=1) * c for _ in range(input_size)]
        else:
            self.weights = weights

        self.last_input: Optional[Vector] = None
        """
        Armazena a última entrada enviada a esse perceptron.
        
        Caso seja `None`, o método ``predict`` ainda não foi chamado.
        """

        self.last_output: Optional[float] = None
        """
        Armazena a última saída produzida por esse perceptron.
        
        Caso seja `None`, o método ``predict`` ainda não foi chamado.
        """

    def predict(self, sample: Vector) -> float:
        """
        Classifica uma amostra de acordo com uma função de ativação.

        :param sample: a amostra a ser prevista.
        :return: a classe dessa amostra, entre 0 e 1.
        """
        assert len(sample) == len(self.weights)
        y: float = 1.0 * self.bias
        y += sum(value * weight for value, weight in zip(sample, self.weights))
        y = self.transform(y)

        self.last_input = sample
        self.last_output = y
        return y

    def transform(self, value: float) -> float:
        """
        Representa a função de ativação desse perceptron.

        :param value: o valor a ser transformado.
        :return: o valor de ativação respectivo.
        """
        return sigmoid(value)

    def update(self, weights: Vector):
        """
        Atualiza os pesos desse perceptron.

        :param weights: os novos pesos.
        """
        self.weights = weights


class InputPerceptron(Perceptron):
    """
    Representa um perceptron de entrada de uma rede neural.

    As propriedades desse perceptron são 1) recebe apenas um valor de entrada e 2) seu bias é zero, seu peso é um e não
    possui função de ativação (isto é, não altera a entrada).
    """

    def __init__(self):
        super().__init__(input_size=1, bias=0.0, weights=[1.0])

    def transform(self, value: float) -> float:
        return value


class UpdatingPerceptron(Perceptron):
    """
    Representa um perceptron de uma rede neural cujos pesos estão sendo atualizados.
    """

    def __init__(self, perceptron: Perceptron):
        super().__init__(len(perceptron.weights), perceptron.bias, weights=perceptron.weights)
        self._perceptron: Perceptron = perceptron

        self.last_input = perceptron.last_input
        self.last_output = perceptron.last_output

        self.gradients: List[Gradient] = []
        """
        Os gradientes desse perceptron.
        
        Cada gradiente está associado com um peso. Caso o gradiente `δe` de um peso seja calculado, é possível calcular
        a variação de peso que esse gradiente causa no peso original por meio da fórmula `- ε * δe.δw`, onde `ε` é a
        taxa de aprendizagem do perceptron.
        """

    def put(self, gradient: Gradient):
        """
        Adiciona um gradiente a esse perceptron.

        Na `i`-ésima chamada desse método, o gradiente a ser adicionado estará associado com o peso `i` em ``weights``.

        :param gradient: o gradiente a ser adicionado.
        """
        self.gradients.append(gradient)

    def update(self, weights: Vector):
        self._perceptron.update(weights)
