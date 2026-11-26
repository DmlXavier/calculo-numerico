"""Programa principal: Aplicação do Método dos Mínimos Quadrados (MMQ).

Contexto prático (Ciência da Computação):
    Uma empresa possui um sistema de pesquisa de informações de clientes e
    realizou testes de desempenho com diferentes quantidades de registros.
    O objetivo é ajustar um modelo linear (y = a*x + b) pelo Método dos
    Mínimos Quadrados para estimar o tempo de execução quando a base crescer
    para 100.000 registros.

Exemplos de uso:
    python main.py
    python main.py --prever 100000 --sem-exibir
    python main.py --interativo
"""

import argparse
import sys
from pathlib import Path
from typing import Sequence

from mmq import ResultadoAjusteLinear, ajustar_reta_mmq
from visualizacao import gerar_grafico_mmq

# Dados hipotéticos de benchmark coletados nos 5 testes descritos no trabalho:
# x -> quantidade de registros pesquisados
# y -> tempo de execução medido em milissegundos (ms)
REGISTROS_TESTADOS: list[int] = [1_000, 5_000, 10_000, 20_000, 50_000]
TEMPOS_MEDIDOS_MS: list[float] = [2.0, 9.0, 19.0, 38.0, 95.0]
PREVISAO_PADRAO_REGISTROS: int = 100_000


def formatar_br(valor: float, casas: int = 4) -> str:
    """Formata um número no padrão brasileiro (milhar com '.' e decimal com ',')."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def imprimir_relatorio(
    valores_x: Sequence[float],
    valores_y: Sequence[float],
    resultado: ResultadoAjusteLinear,
    x_previsao: float,
) -> None:
    """Exibe no terminal o passo a passo matemático, os resíduos e a previsão."""
    linha = "=" * 76
    sublinha = "-" * 76

    print(linha)
    print("  MÉTODO DOS MÍNIMOS QUADRADOS — PREVISÃO DE DESEMPENHO DE ALGORITMO")
    print(linha)

    print("\n1. DADOS COLETADOS NOS TESTES")
    print(sublinha)
    print(f"{'Teste':>6} | {'Registros (x)':>16} | {'Tempo real (y em ms)':>22}")
    print(sublinha)
    for indice, (x, y) in enumerate(zip(valores_x, valores_y), start=1):
        print(f"{indice:>6} | {formatar_br(x, 0):>16} | {formatar_br(y, 2):>22}")

    print("\n2. SOMATÓRIOS DAS EQUAÇÕES NORMAIS DO MMQ")
    print(sublinha)
    print(f"  n (quantidade de pontos) = {resultado.n}")
    print(f"  Σx   = {formatar_br(resultado.soma_x, 0)}")
    print(f"  Σy   = {formatar_br(resultado.soma_y, 2)}")
    print(f"  Σx²  = {formatar_br(resultado.soma_x2, 0)}")
    print(f"  Σxy  = {formatar_br(resultado.soma_xy, 2)}")
    print(f"  x̄    = {formatar_br(resultado.media_x, 2)} registros")
    print(f"  ȳ    = {formatar_br(resultado.media_y, 2)} ms")

    print("\n3. EQUAÇÃO DA RETA DE MELHOR AJUSTE (y = ax + b)")
    print(sublinha)
    print(f"  Inclinação (a)  = {formatar_br(resultado.a, 7)} ms por registro")
    print(f"  Intercepto (b)  = {formatar_br(resultado.b, 4)} ms")
    print(f"  Equação obtida  : {resultado.equacao_formatada()}")
    print(
        "\n  Interpretação:"
        f"\n  - Cada registro adicional acrescenta cerca de "
        f"{formatar_br(resultado.a, 7)} ms (~{formatar_br(resultado.a * 1000, 4)} ms a cada 1.000 registros)."
        "\n  - O coeficiente 'b' é um ajuste matemático da reta e não representa "
        "necessariamente um tempo físico real para 0 registros."
    )

    print("\n4. ANÁLISE DOS RESÍDUOS (ERROS = y_real - y_estimado)")
    print(sublinha)
    print(
        f"{'Registros (x)':>14} | {'y real':>10} | {'y estimado':>12} | "
        f"{'Resíduo (e)':>13} | {'Resíduo² (e²)':>14}"
    )
    print(sublinha)
    for x, y_real, y_est, res, res2 in zip(
        valores_x,
        valores_y,
        resultado.y_estimados,
        resultado.residuos,
        resultado.residuos_quadrados,
    ):
        print(
            f"{formatar_br(x, 0):>14} | "
            f"{formatar_br(y_real, 2):>10} | "
            f"{formatar_br(y_est, 4):>12} | "
            f"{formatar_br(res, 4):>13} | "
            f"{formatar_br(res2, 6):>14}"
        )
    print(sublinha)
    print(
        f"  Soma dos Quadrados dos Resíduos (SQR) = "
        f"{formatar_br(resultado.soma_quadrados_residuos, 6)}"
    )
    print(
        f"  Coeficiente de Determinação (R²)      = "
        f"{formatar_br(resultado.r_quadrado, 6)} "
        f"({formatar_br(resultado.r_quadrado * 100, 4)}%)"
    )

    tempo_previsto = resultado.prever(x_previsao)
    print("\n5. PREVISÃO COM O MODELO AJUSTADO")
    print(sublinha)
    print(f"  Entrada desejada : {formatar_br(x_previsao, 0)} registros")
    print(
        f"  Cálculo          : y = {formatar_br(resultado.a, 7)} × "
        f"{formatar_br(x_previsao, 0)} {'-' if resultado.b < 0 else '+'} "
        f"{formatar_br(abs(resultado.b), 4)}"
    )
    print(f"  Tempo estimado   : {formatar_br(tempo_previsto, 2)} ms")
    print(
        "\n  Observação: trata-se de uma estimativa matemática baseada na tendência "
        "linear dos\n  testes. Em produção, memória, CPU, cache, linguagem e "
        "complexidade real do algoritmo\n  podem alterar o tempo efetivo."
    )
    print(linha)


def executar_modo_interativo(resultado: ResultadoAjusteLinear) -> None:
    """Permite consultar previsões para novas quantidades de registros no terminal."""
    print("\n--- MODO INTERATIVO DE PREVISÃO ---")
    print("Digite uma quantidade de registros para estimar o tempo (ou 'sair').")

    while True:
        entrada = input("\nQuantidade de registros: ").strip().lower()
        if entrada in {"sair", "q", "quit", ""}:
            print("Encerrando modo interativo.")
            break

        try:
            # Aceita entradas como '100000', '100.000' ou '100_000'
            normalizado = entrada.replace(".", "").replace("_", "").replace(",", ".")
            registros = float(normalizado)
            if registros < 0:
                print("Informe um número não negativo de registros.")
                continue
        except ValueError:
            print("Entrada inválida. Digite apenas números (ex.: 150000) ou 'sair'.")
            continue

        estimativa = resultado.prever(registros)
        print(
            f"  -> Tempo estimado para {formatar_br(registros, 0)} registros: "
            f"{formatar_br(estimativa, 2)} ms"
        )


def criar_parser() -> argparse.ArgumentParser:
    """Configura os argumentos de linha de comando do programa."""
    parser = argparse.ArgumentParser(
        description=(
            "Aplica o Método dos Mínimos Quadrados para prever o tempo de "
            "execução de um algoritmo conforme a quantidade de registros."
        )
    )
    parser.add_argument(
        "--prever",
        type=float,
        default=PREVISAO_PADRAO_REGISTROS,
        help="Quantidade de registros para previsão (padrão: 100000).",
    )
    parser.add_argument(
        "--salvar-grafico",
        type=Path,
        default=Path("grafico_mmq.png"),
        help="Caminho do arquivo PNG para salvar o gráfico (padrão: grafico_mmq.png).",
    )
    parser.add_argument(
        "--sem-grafico",
        action="store_true",
        help="Executa apenas o cálculo numérico sem gerar imagem nem abrir janela.",
    )
    parser.add_argument(
        "--sem-exibir",
        action="store_true",
        help="Salva o gráfico em arquivo sem abrir a janela interativa.",
    )
    parser.add_argument(
        "--interativo",
        action="store_true",
        help="Após o relatório, permite consultar previsões para outros valores de x.",
    )
    return parser


def main() -> None:
    """Ponto de entrada da aplicação."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = criar_parser().parse_args()

    # 1. Aplica o Método dos Mínimos Quadrados aos dados de benchmark
    resultado = ajustar_reta_mmq(REGISTROS_TESTADOS, TEMPOS_MEDIDOS_MS)

    # 2. Mostra todo o desenvolvimento matemático e a previsão no terminal
    imprimir_relatorio(
        valores_x=REGISTROS_TESTADOS,
        valores_y=TEMPOS_MEDIDOS_MS,
        resultado=resultado,
        x_previsao=args.prever,
    )

    # 3. Gera o gráfico com os pontos reais, a reta ajustada e os resíduos
    if not args.sem_grafico:
        try:
            caminho_salvo = gerar_grafico_mmq(
                valores_x=REGISTROS_TESTADOS,
                valores_y=TEMPOS_MEDIDOS_MS,
                resultado=resultado,
                x_previsao=args.prever,
                caminho_saida=args.salvar_grafico,
                exibir=not args.sem_exibir,
            )
            if caminho_salvo is not None:
                print(f"\nGráfico salvo em: {caminho_salvo.resolve()}")
        except RuntimeError as erro:
            print(f"\n[Aviso] Não foi possível gerar o gráfico: {erro}")

    # 4. Opcionalmente abre consultas livres no terminal
    if args.interativo:
        executar_modo_interativo(resultado)


if __name__ == "__main__":
    main()
