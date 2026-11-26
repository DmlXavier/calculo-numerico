"""Testes unitários para validar a implementação do Método dos Mínimos Quadrados."""

import unittest

from main import PREVISAO_PADRAO_REGISTROS, REGISTROS_TESTADOS, TEMPOS_MEDIDOS_MS
from mmq import ajustar_reta_mmq


class TestMetodoMinimosQuadrados(unittest.TestCase):
    """Valida cálculos exatos, propriedades matemáticas e casos de borda."""

    def test_exemplo_documento_desempenho_algoritmo(self) -> None:
        """Verifica se os resultados batem com o exemplo numérico do documento."""
        resultado = ajustar_reta_mmq(REGISTROS_TESTADOS, TEMPOS_MEDIDOS_MS)

        self.assertEqual(resultado.n, 5)
        self.assertEqual(resultado.soma_x, 86_000.0)
        self.assertEqual(resultado.soma_y, 163.0)
        self.assertEqual(resultado.soma_x2, 3_026_000_000.0)
        self.assertEqual(resultado.soma_xy, 5_747_000.0)

        # Valores esperados no documento: a ≈ 0,0019029, b ≈ -0,1298, y(100k) ≈ 190,16
        self.assertAlmostEqual(resultado.a, 0.0019029, places=7)
        self.assertAlmostEqual(resultado.b, -0.1298, places=4)
        self.assertAlmostEqual(
            resultado.prever(PREVISAO_PADRAO_REGISTROS), 190.16, places=2
        )

        # Propriedade matemática do MMQ com intercepto: soma dos resíduos == 0
        self.assertAlmostEqual(sum(resultado.residuos), 0.0, places=9)
        self.assertGreater(resultado.r_quadrado, 0.9999)

    def test_reta_perfeita_sem_erro(self) -> None:
        """Pontos perfeitamente alinhados em y = 3x + 5 devem gerar SQR = 0 e R² = 1."""
        xs = [0, 1, 2, 3, 4]
        ys = [3 * x + 5 for x in xs]
        resultado = ajustar_reta_mmq(xs, ys)

        self.assertAlmostEqual(resultado.a, 3.0, places=12)
        self.assertAlmostEqual(resultado.b, 5.0, places=12)
        self.assertAlmostEqual(resultado.soma_quadrados_residuos, 0.0, places=12)
        self.assertAlmostEqual(resultado.r_quadrado, 1.0, places=12)

    def test_validacao_tamanhos_diferentes(self) -> None:
        """Deve lançar ValueError quando x e y têm quantidades diferentes."""
        with self.assertRaises(ValueError):
            ajustar_reta_mmq([1, 2, 3], [1, 2])

    def test_validacao_poucos_pontos(self) -> None:
        """Deve lançar ValueError com menos de 2 pontos."""
        with self.assertRaises(ValueError):
            ajustar_reta_mmq([10], [5])

    def test_validacao_x_constante(self) -> None:
        """Deve lançar ValueError quando todos os valores de x são iguais."""
        with self.assertRaises(ValueError):
            ajustar_reta_mmq([5, 5, 5], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
