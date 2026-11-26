"""Módulo de implementação matemática do Método dos Mínimos Quadrados (MMQ).

Este módulo implementa do zero (sem depender de bibliotecas externas) o ajuste
linear simples pelo Método dos Mínimos Quadrados para encontrar a reta de
melhor ajuste:

    y = a * x + b

onde:
    - a (coeficiente angular): inclinação da reta;
    - b (coeficiente linear): intercepto no eixo vertical (y quando x = 0).

Os coeficientes são obtidos minimizando a Soma dos Quadrados dos Resíduos (SQR):

    S(a, b) = sum((y_i - (a * x_i + b)) ** 2)
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ResultadoAjusteLinear:
    """Armazena os coeficientes, somatórios intermediários e métricas do ajuste."""

    n: int
    soma_x: float
    soma_y: float
    soma_x2: float
    soma_xy: float
    media_x: float
    media_y: float
    a: float
    b: float
    y_estimados: list[float]
    residuos: list[float]
    residuos_quadrados: list[float]
    soma_quadrados_residuos: float
    soma_quadrados_total: float
    r_quadrado: float

    def prever(self, x: float) -> float:
        """Estima o valor de y para um dado valor de x usando y = a*x + b."""
        return prever_linear(self.a, self.b, x)

    def equacao_formatada(self, separador_decimal: str = ",") -> str:
        """Retorna a equação da reta formatada para exibição didática."""
        sinal_b = "-" if self.b < 0 else "+"
        a_str = f"{self.a:.7f}"
        b_str = f"{abs(self.b):.4f}"
        if separador_decimal != ".":
            a_str = a_str.replace(".", separador_decimal)
            b_str = b_str.replace(".", separador_decimal)
        return f"y = {a_str}x {sinal_b} {b_str}"


def prever_linear(a: float, b: float, x: float) -> float:
    """Calcula a previsão linear y = a * x + b para uma entrada x."""
    return a * x + b


def ajustar_reta_mmq(
    valores_x: Sequence[float],
    valores_y: Sequence[float],
) -> ResultadoAjusteLinear:
    """Calcula a reta de melhor ajuste y = a*x + b pelo Método dos Mínimos Quadrados.

    Passos matemáticos aplicados:
        1. Validação dos dados de entrada.
        2. Cálculo dos somatórios: sum(x), sum(y), sum(x^2) e sum(x*y).
        3. Resolução das equações normais para obter 'a' e 'b':
               a = (n * sum(x*y) - sum(x) * sum(y)) / (n * sum(x^2) - (sum(x))^2)
               b = (sum(y) - a * sum(x)) / n
        4. Cálculo dos valores estimados (y_hat), dos resíduos (e_i = y_i - y_hat_i),
           da Soma dos Quadrados dos Resíduos (SQR) e do coeficiente de
           determinação (R^2).

    Args:
        valores_x: Sequência com os valores da variável independente (ex.: registros).
        valores_y: Sequência com os valores da variável dependente (ex.: tempo em ms).

    Returns:
        Instância de ResultadoAjusteLinear contendo coeficientes e métricas.

    Raises:
        ValueError: Se as listas tiverem tamanhos diferentes, menos de 2 pontos,
            ou se todos os valores de x forem iguais (variância nula).
    """
    if len(valores_x) != len(valores_y):
        raise ValueError(
            "As listas de valores x e y devem ter a mesma quantidade de elementos."
        )

    n = len(valores_x)
    if n < 2:
        raise ValueError(
            "São necessários pelo menos 2 pontos para ajustar uma reta."
        )

    xs = [float(x) for x in valores_x]
    ys = [float(y) for y in valores_y]

    # 1. Cálculo dos somatórios exigidos pelas equações normais do MMQ
    soma_x = sum(xs)
    soma_y = sum(ys)
    soma_x2 = sum(x * x for x in xs)
    soma_xy = sum(x * y for x, y in zip(xs, ys))

    media_x = soma_x / n
    media_y = soma_y / n

    # 2. Verificação do denominador (evita divisão por zero quando todos x são iguais)
    denominador = n * soma_x2 - (soma_x ** 2)
    if denominador == 0:
        raise ValueError(
            "Não é possível ajustar uma reta: todos os valores de x são iguais."
        )

    # 3. Obtenção dos coeficientes angular (a) e linear (b)
    numerador = n * soma_xy - soma_x * soma_y
    a = numerador / denominador
    b = media_y - a * media_x

    # 4. Avaliação do ajuste: valores estimados, resíduos e SQR
    y_estimados = [prever_linear(a, b, x) for x in xs]
    residuos = [y_real - y_est for y_real, y_est in zip(ys, y_estimados)]
    residuos_quadrados = [erro ** 2 for erro in residuos]
    soma_quadrados_residuos = sum(residuos_quadrados)

    # 5. Coeficiente de determinação (R^2): mede o quão bem a reta explica os dados
    soma_quadrados_total = sum((y - media_y) ** 2 for y in ys)
    if soma_quadrados_total == 0:
        r_quadrado = 1.0
    else:
        r_quadrado = 1.0 - (soma_quadrados_residuos / soma_quadrados_total)

    return ResultadoAjusteLinear(
        n=n,
        soma_x=soma_x,
        soma_y=soma_y,
        soma_x2=soma_x2,
        soma_xy=soma_xy,
        media_x=media_x,
        media_y=media_y,
        a=a,
        b=b,
        y_estimados=y_estimados,
        residuos=residuos,
        residuos_quadrados=residuos_quadrados,
        soma_quadrados_residuos=soma_quadrados_residuos,
        soma_quadrados_total=soma_quadrados_total,
        r_quadrado=r_quadrado,
    )
