import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
from utils.dataLoader import TEXT_COLUMN, TARGET_COLUMN

def applyStyle():
    plt.style.use('default')
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 3
    
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.edgecolor'] = 'black'
    plt.rcParams['legend.fancybox'] = False 

applyStyle()

COLOR_TARGET = "#5d54a4"
COLOR_PRED = "#9154a4"
COLOR_ERR = '#404040'
COLOR_ATTRACTOR = "#892BC0"

SERIES_COLORS = [COLOR_TARGET, COLOR_PRED, COLOR_ATTRACTOR, COLOR_ERR, 'black']
MATRIX_CMAP = LinearSegmentedColormap.from_list('rcPurples', ['white', COLOR_TARGET])

def _categoricalAxis(ax):
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)

def plotClassDistribution(df, title=None):
    counts = df[TARGET_COLUMN].value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))
    
    ax.bar(counts.index, counts.values, color=COLOR_TARGET, alpha=0.7, edgecolor='black', linewidth=0.5)

    for i, value in enumerate(counts.values):
        ax.text(i, value, f'{value / counts.sum():.1%}', ha='center', va='bottom', fontsize=10)

    if title:
        ax.set_title(title)

    ax.set_xlabel("Party")
    ax.set_ylabel("Tweets")
    ax.grid(True, axis='y', linestyle=':', alpha=0.3)
    _categoricalAxis(ax)
    
    plt.show()

def plotLengthDistribution(df, unit='words', title=None):
    lengths = df[TEXT_COLUMN].str.split().str.len() if unit == 'words' else df[TEXT_COLUMN].str.len()
    bins = np.arange(0, lengths.max() + 2)

    fig, ax = plt.subplots(figsize=(10, 4))
    
    for color, (label, group) in zip(SERIES_COLORS, lengths.groupby(df[TARGET_COLUMN])):
        ax.hist(group, bins=bins, density=True, histtype='step', color=color, linewidth=1.0, label=label)

    if title:
        ax.set_title(title)

    ax.set_xlabel(f"Tweet Length ({unit})")
    ax.set_ylabel("Density")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.3)
    
    plt.show()

def plotNoiseStats(noiseSummary, title=None):
    data = noiseSummary.drop(index='all')

    fig, ax = plt.subplots(figsize=(10, 4))
    
    data.T.plot.bar(ax=ax, color=SERIES_COLORS[:len(data)], alpha=0.7, edgecolor='black', linewidth=0.5, width=0.8)

    if title:
        ax.set_title(title)

    ax.set_ylabel("Share of Tweets")
    ax.set_xticklabels(data.columns, rotation=0)
    ax.legend(ncol=len(data), loc='upper center', bbox_to_anchor=(0.5, 1.15))
    ax.grid(True, axis='y', linestyle=':', alpha=0.3)
    _categoricalAxis(ax)
    
    plt.show()

def plotConfusionMatrix(actuals, predictions, normalize=True, title=None):
    labels = sorted(set(actuals))
    matrix = confusion_matrix(actuals, predictions, labels=labels, normalize='true' if normalize else None)

    fig, ax = plt.subplots(figsize=(6, 5))
    
    ax.imshow(matrix, cmap=MATRIX_CMAP, vmin=0)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i, j]
            text = f'{value:.2f}' if normalize else f'{value:d}'
            ax.text(j, i, text, ha='center', va='center', fontsize=10, color='white' if value > matrix.max() / 2 else 'black')

    if title:
        ax.set_title(title)

    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(which='minor', bottom=False, top=False, left=False, right=False)
    
    plt.show()

def plotF1PerClass(metricsByModel, title=None):
    data = pd.DataFrame({name: metrics['f1PerClass'] for name, metrics in metricsByModel.items()})

    fig, ax = plt.subplots(figsize=(10, 4))
    
    data.plot.bar(ax=ax, color=SERIES_COLORS[:len(data.columns)], alpha=0.7, edgecolor='black', linewidth=0.5, width=0.8)

    if title:
        ax.set_title(title)

    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(data.index, rotation=0)
    ax.legend(loc='lower right', framealpha=1)
    ax.grid(True, axis='y', linestyle=':', alpha=0.3)
    _categoricalAxis(ax)
    
    plt.show()

def _shadeGap(ax, x, trainScores, validationScores):
    ax.fill_between(x, validationScores, trainScores, color=COLOR_ERR, alpha=0.12, linewidth=0, label='Generalization Gap')

def plotTrainingHistory(history, metric='loss', title=None):
    trainValues = history.get(metric)
    validationValues = history.get(f'val_{metric}')
    epochs = np.arange(1, len(validationValues if validationValues is not None else trainValues) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    if trainValues is not None:
        ax.plot(epochs, trainValues, color='black', marker='o', markersize=4, linewidth=0.8, label='Train')
    if validationValues is not None:
        ax.plot(epochs, validationValues, color=COLOR_PRED, marker='s', markersize=4, linestyle='--', linewidth=1.5, label='Validation')
        if trainValues is not None:
            _shadeGap(ax, epochs, trainValues, validationValues)

        bestEpoch = int(np.argmin(validationValues) if 'loss' in metric else np.argmax(validationValues)) + 1
        ax.axvline(bestEpoch, color=COLOR_ATTRACTOR, linestyle=':', linewidth=1.0, label=f'best epoch = {bestEpoch}')

    if title:
        ax.set_title(title)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.legend(loc='best')
    ax.grid(True, linestyle=':', alpha=0.3)

    plt.show()
