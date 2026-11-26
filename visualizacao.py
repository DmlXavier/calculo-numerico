"""Módulo de visualização gráfica do Método dos Mínimos Quadrados.

Gera um gráfico didático contendo:
    1. Painel principal: pontos reais coletados, reta ajustada (destacando o
       trecho observado e o trecho de extrapolação) e o ponto previsto.
    2. Painel inferior: resíduos (erros = valor real - valor estimado) de cada
       teste, ilustrando o que o Método dos Mínimos Quadrados minimiza.
"""

from pathlib import Path
from typing import Optional, Sequence

from mmq import ResultadoAjusteLinear


def _formatar_inteiro_br(valor: float, _pos: int = 0) -> str:
    """Formata inteiros no padrão brasileiro (ex.: 100000 -> '100.000')."""
    return f"{int(round(valor)):,}".replace(",", ".")


def _formatar_decimal_br(valor: float, casas: int = 2) -> str:
    """Formata decimais no padrão brasileiro (ex.: 190.16 -> '190,16')."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def gerar_grafico_mmq(
    valores_x: Sequence[float],
    valores_y: Sequence[float],
    resultado: ResultadoAjusteLinear,
    x_previsao: Optional[float] = None,
    caminho_saida: Optional[Path] = None,
    exibir: bool = True,
) -> Optional[Path]:
    """Cria o gráfico com os dados observados, a reta ajustada e os resíduos.

    Args:
        valores_x: Valores observados de entrada (quantidade de registros).
        valores_y: Valores observados de saída (tempo de execução em ms).
        resultado: Objeto retornado por `ajustar_reta_mmq`.
        x_previsao: Ponto opcional de previsão (ex.: 100_000 registros).
        caminho_saida: Caminho do arquivo PNG para salvar a figura.
        exibir: Se True, abre a janela interativa do matplotlib.

    Returns:
        O caminho do arquivo salvo (se `caminho_saida` for informado), ou None.
    """
    try:
        import matplotlib

        if not exibir:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError as erro:
        raise RuntimeError(
            "A biblioteca 'matplotlib' não está instalada. "
            "Instale-a com: pip install -r requirements.txt"
        ) from erro

    xs = [float(x) for x in valores_x]
    ys = [float(y) for y in valores_y]
    x_min = min(xs)
    x_max = max(xs)

    fig, (ax_principal, ax_residuos) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 7.2),
        gridspec_kw={"height_ratios": [3.2, 1.3]},
        constrained_layout=True,
    )

    # --- Painel 1: Dados coletados, reta ajustada e previsão ---
    # Segmento da reta dentro da faixa observada (interpolação)
    reta_obs_x = [0.0, x_max]
    reta_obs_y = [resultado.prever(x) for x in reta_obs_x]
    ax_principal.plot(
        reta_obs_x,
        reta_obs_y,
        color="#1f77b4",
        linewidth=2.4,
        label="Reta ajustada (MMQ - faixa observada)",
        zorder=2,
    )

    # Segmento de extrapolação, caso a previsão esteja além dos dados coletados
    if x_previsao is not None and x_previsao > x_max:
        reta_ext_x = [x_max, x_previsao]
        reta_ext_y = [resultado.prever(x) for x in reta_ext_x]
        ax_principal.plot(
            reta_ext_x,
            reta_ext_y,
            color="#1f77b4",
            linewidth=2.2,
            linestyle="--",
            label="Extrapolação do modelo linear",
            zorder=2,
        )

    # Pontos reais coletados nos testes
    ax_principal.scatter(
        xs,
        ys,
        color="#0d3b66",
        s=75,
        edgecolors="white",
        linewidths=1.2,
        label="Dados coletados nos testes",
        zorder=4,
    )

    # Rótulos dos pontos reais com posicionamento alternado para evitar sobreposição
    x_span = max(x_max, x_previsao or x_max) or 1.0
    pares_ordenados = sorted(zip(xs, ys), key=lambda par: par[0])
    for idx, (x, y) in enumerate(pares_ordenados):
        proximo_perto = (
            idx + 1 < len(pares_ordenados)
            and (pares_ordenados[idx + 1][0] - x) < 0.12 * x_span
        )
        anterior_perto = (
            idx > 0 and (x - pares_ordenados[idx - 1][0]) < 0.12 * x_span
        )

        if proximo_perto or anterior_perto:
            # Alterna acima-esquerda e abaixo-direita com linha-guia discreta
            if idx % 2 == 0:
                if idx == 0:
                    offset = (-4, 24)
                    ha = "left"
                else:
                    offset = (0, 32)
                    ha = "center"
            else:
                offset = (22, -14)
                ha = "left"
            seta = {"arrowstyle": "-", "color": "#8d99ae", "lw": 0.9}
        else:
            offset = (-12, 12)
            ha = "right"
            seta = None

        ax_principal.annotate(
            f"({_formatar_inteiro_br(x)}; {_formatar_decimal_br(y, 0)} ms)",
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            fontsize=8.5,
            color="#2b2d42",
            arrowprops=seta,
        )

    # Ponto previsto (ex.: 100.000 registros)
    if x_previsao is not None:
        y_previsao = resultado.prever(x_previsao)
        ax_principal.scatter(
            [x_previsao],
            [y_previsao],
            color="#d62828",
            marker="D",
            s=95,
            edgecolors="white",
            linewidths=1.2,
            label=(
                f"Previsão ({_formatar_inteiro_br(x_previsao)} registros "
                f"≈ {_formatar_decimal_br(y_previsao, 2)} ms)"
            ),
            zorder=5,
        )
        # Linhas-guia pontilhadas até os eixos
        ax_principal.vlines(
            x_previsao, 0, y_previsao, colors="#d62828", linestyles=":", alpha=0.65
        )
        ax_principal.hlines(
            y_previsao, 0, x_previsao, colors="#d62828", linestyles=":", alpha=0.65
        )
        ax_principal.annotate(
            (
                f"Previsão:\n{_formatar_inteiro_br(x_previsao)} registros\n"
                f"≈ {_formatar_decimal_br(y_previsao, 2)} ms"
            ),
            xy=(x_previsao, y_previsao),
            xytext=(-205, -6),
            textcoords="offset points",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#9b2226",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#fff5f5",
                "edgecolor": "#d62828",
                "alpha": 0.96,
            },
            arrowprops={"arrowstyle": "->", "color": "#d62828", "lw": 1.4},
            zorder=6,
        )

    # Caixa de resumo matemático posicionada na região livre à esquerda
    caixa_info = (
        f"Equação: {resultado.equacao_formatada()}\n"
        f"SQR (Soma dos Erros²): {_formatar_decimal_br(resultado.soma_quadrados_residuos, 4)}\n"
        f"R² (Qualidade do ajuste): {_formatar_decimal_br(resultado.r_quadrado * 100, 4)}%"
    )
    ax_principal.text(
        0.02,
        0.79,
        caixa_info,
        transform=ax_principal.transAxes,
        verticalalignment="top",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#f8f9fa",
            "edgecolor": "#adb5bd",
            "alpha": 0.96,
        },
        zorder=6,
    )

    limite_x_sup = max(x_max, x_previsao or x_max) * 1.07
    limite_y_sup = max(
        max(ys),
        resultado.prever(x_previsao) if x_previsao is not None else max(ys),
    ) * 1.16

    ax_principal.set_xlim(-limite_x_sup * 0.03, limite_x_sup)
    ax_principal.set_ylim(-limite_y_sup * 0.08, limite_y_sup)
    ax_principal.xaxis.set_major_formatter(FuncFormatter(_formatar_inteiro_br))
    ax_principal.set_title(
        "Método dos Mínimos Quadrados: Previsão do Tempo de Execução de Algoritmo",
        fontsize=12.5,
        fontweight="bold",
        pad=10,
    )
    ax_principal.set_xlabel("Quantidade de registros (x)", fontsize=10.5)
    ax_principal.set_ylabel("Tempo de execução em ms (y)", fontsize=10.5)
    ax_principal.grid(True, linestyle="--", alpha=0.35)
    ax_principal.legend(loc="lower right", frameon=True, fontsize=9)

    # --- Painel 2: Resíduos (erros) em cada teste observado ---
    posicoes = list(range(len(xs)))
    rotulos_x = [_formatar_inteiro_br(x) for x in xs]
    cores_barras = [
        "#2a9d8f" if erro >= 0 else "#e76f51" for erro in resultado.residuos
    ]

    barras = ax_residuos.bar(
        posicoes,
        resultado.residuos,
        width=0.52,
        color=cores_barras,
        edgecolor="#2b2d42",
        linewidth=0.8,
        zorder=3,
    )
    ax_residuos.axhline(0, color="#2b2d42", linewidth=1.0, zorder=2)

    max_abs_residuo = max(abs(r) for r in resultado.residuos) or 0.1
    ax_residuos.set_ylim(-max_abs_residuo * 1.55, max_abs_residuo * 1.55)

    for barra, erro in zip(barras, resultado.residuos):
        deslocamento_y = 4 if erro >= 0 else -12
        ax_residuos.annotate(
            f"{'+' if erro >= 0 else ''}{_formatar_decimal_br(erro, 3)} ms",
            xy=(barra.get_x() + barra.get_width() / 2, erro),
            xytext=(0, deslocamento_y),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#2b2d42",
        )

    ax_residuos.set_xticks(posicoes)
    ax_residuos.set_xticklabels(rotulos_x)
    ax_residuos.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _p: _formatar_decimal_br(v, 2))
    )
    ax_residuos.set_title(
        "Resíduos por teste (eᵢ = y_real − y_estimado)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_residuos.set_xlabel("Quantidade de registros testados", fontsize=9.5)
    ax_residuos.set_ylabel("Resíduo (ms)", fontsize=9.5)
    ax_residuos.grid(True, axis="y", linestyle="--", alpha=0.35, zorder=0)

    if caminho_saida is not None:
        caminho_saida = Path(caminho_saida)
        fig.savefig(caminho_saida, dpi=150)

    if exibir:
        plt.show()
    else:
        plt.close(fig)

    return caminho_saida
