from typing import Iterable, Optional, Tuple
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

def plot_candlestick(
    df: pd.DataFrame,
    n: int = 1,
    style: str = 'yahoo',
    volume: bool = True,
    title: Optional[str] = None,
    figratio: Tuple[int, int] = (16, 9),
    figscale: float = 1.0,
) -> None:

    if n <= 1:
        df_plot = df.copy()
    else:
        #Group rows in chunks of n
        groups = np.arange(len(df)) // n

        #Arguments to select each specific value
        agg_funcs = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        #Apply the arguments to the groups of 'n' days
        df_plot = df.groupby(groups).agg(agg_funcs)

        #Use the last date in each group as the new index
        df_plot.index = df.groupby(groups).apply(lambda g: g.index[-1])

        #Convert to float for mplfinance compatibility
        df_plot = df_plot.astype(float)

    #Plot the candlestick chart
    mpf.plot(
        df_plot,
        type='candle',
        style=style,
        volume=volume,
        title=title,
        figratio=figratio,
        figscale=figscale,
    )


def plot_boxplot(
    df: pd.DataFrame,
    n: int = 1,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    columns: Iterable[str] = ("open", "high", "low", "close"),
) -> None:

    if n <= 1:
        grouped_dfs = [df]
    else:
        #Split DataFrame into chunks of size 'n'
        groups = np.arange(len(df)) // n
        grouped_dfs = [g for _, g in df.groupby(groups) if len(g) == n]

    #Create the subplot
    ax = plt.subplots(figsize=figsize)

    #Lists to hold data related to the boxplots
    data_to_plot = []
    x_labels = []

    for i, group in enumerate(grouped_dfs):
        selected = group[list(columns)].values.flatten()    #Extract only relevant columns
        data_to_plot.append(selected)
        x_labels.append(f"{group.index[0].date()} – {group.index[-1].date()}")

    #Plot boxplot chart
    ax.boxplot(data_to_plot, patch_artist=True)
    ax.set_title(title)
    ax.set_xlabel("Trading Window")
    ax.set_ylabel("Value")
    ax.set_xticklabels(x_labels, rotation=45)
    plt.show()
