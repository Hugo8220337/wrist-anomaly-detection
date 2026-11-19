from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

def plot_movement_data(
    clean_data, outlier_data, algorithm_name, target_movement, model_representation,
    other_data=None, other_data_legend_label=None, axis_names=['x', 'y', 'z'], save_fig=False, save_path=None
):
    """
    Plots movement data for both clean and outlier samples, compatible with DTW and Feature Extraction models.

    Parameters:
    - clean_data: np.ndarray or pd.DataFrame
        Clean data (shape: (n_samples, T, n_features) for DTW or DataFrame for Feature Extraction).
    - outlier_data: np.ndarray or pd.DataFrame
        Outlier data (same format as clean_data).
    - algorithm_name: str
        Name of the algorithm used for outlier detection (e.g., 'DTW', 'DBSCAN').
    - target_movement: str
        The movement type (e.g., 'Move', 'Pick', 'Place').
    - model_representation: str
        The representation of the model (e.g., 'DTW', 'FeatureBased').
    - other_data: np.ndarray or pd.DataFrame, optional
        Additional data to plot (same format as clean_data).
    - axis_names: list
        Names of the axes to plot (default: ['x', 'y', 'z']).
    """
    fig, axs = plt.subplots(1, len(axis_names), figsize=(18, 6))

    # Define legend
    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, alpha=0.6, label='Clean'),
        Line2D([0], [0], color='red', lw=2, alpha=0.9, label='Outlier'),
    ]
    # Add other data to legend if provided
    if other_data_legend_label is not None:
        legend_elements.append(Line2D([0], [0], color='blue', lw=2, alpha=0.6, label=other_data_legend_label))

    for i, axis in enumerate(axis_names):
        ax = axs[i]

        # Plot for DTW-based models (numpy arrays)
        if isinstance(clean_data, np.ndarray) and isinstance(outlier_data, np.ndarray):
            for series in clean_data:
                ax.plot(series[:, i], color='gray', alpha=0.3)
            for series in outlier_data:
                ax.plot(series[:, i], color='red', alpha=0.8, linewidth=2)

        # Plot for Feature Extraction models (DataFrames)
        elif isinstance(clean_data, pd.DataFrame) and isinstance(outlier_data, pd.DataFrame):
            for _, group in clean_data.groupby('temp_key'):
                ax.plot(group['step'], group[axis], color='gray', alpha=0.1)
            for _, group in outlier_data.groupby('temp_key'):
                ax.plot(group['step'], group[axis], color='red', alpha=0.6, linewidth=1.5)

        # Plot other data if provided
        if other_data is not None:
            if isinstance(other_data, np.ndarray):
                for series in other_data:
                    ax.plot(series[:, i], color='blue', alpha=0.3)
                legend_elements.append(Line2D([0], [0], color='blue', lw=5, alpha=1, label=other_data_legend_label))
            elif isinstance(other_data, pd.DataFrame):
                for _, group in other_data.groupby('temp_key'):
                    ax.plot(group['step'], group[axis], color='blue', alpha=0.1)
                legend_elements.append(Line2D([0], [0], color='blue', lw=5, alpha=1, label=other_data_legend_label))

        # Configure subplot
        ax.set_title(f'{axis.capitalize()} Axis ')
        ax.set_xlabel('Step')
        ax.set_ylabel(f'Coordinate {axis}')
        ax.grid(True, linestyle=':', alpha=0.6)

        # Add legend to the last plot
        if i == len(axis_names) - 1:
            ax.legend(handles=legend_elements, loc='upper right')

    # General title
    plt.suptitle(f'Movement Analysis ({algorithm_name} - {model_representation}): {target_movement} - Clean vs Outliers', fontsize=16)
    plt.tight_layout()

    if save_fig and save_path is not None:
        save_dir = Path(save_path)
        file_name = f"{algorithm_name}_{target_movement}_{model_representation}_movement_analysis.png"
        output_file = save_dir / file_name
        plt.savefig(output_file, dpi=300)

    plt.show()


def plot_movement_data_3d(clean_data, outlier_data, target_movement, axis_names=['x', 'y', 'z']):
    """
    Desenha as trajetórias em 3D. 
    Nota: Assume que as colunas ou índices do array correspondem a X, Y e Z nessa ordem.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, alpha=0.6, label='Clean'),
        Line2D([0], [0], color='red', lw=2, alpha=0.9, label='Outlier'),
    ]

    # --- Plot para matrizes NumPy (Modelos baseados em DTW/PCA/UMAP) ---
    if isinstance(clean_data, np.ndarray) and isinstance(outlier_data, np.ndarray):
        # clean_data[:, :, 0] -> X | [:, :, 1] -> Y | [:, :, 2] -> Z
        for series in clean_data:
            ax.plot(series[:, 0], series[:, 1], series[:, 2], color='gray', alpha=0.2)
        for series in outlier_data:
            ax.plot(series[:, 0], series[:, 1], series[:, 2], color='red', alpha=0.8, linewidth=2)

    # --- Plot para DataFrames Pandas (Modelos de Extração de Features) ---
    elif isinstance(clean_data, pd.DataFrame) and isinstance(outlier_data, pd.DataFrame):
        # Extrair os nomes exatos das colunas (ex: 'xw_L', 'yw_L', 'zw_L')
        col_x, col_y, col_z = axis_names[0], axis_names[1], axis_names[2]
        
        for _, group in clean_data.groupby('temp_key'):
            ax.plot(group[col_x], group[col_y], group[col_z], color='gray', alpha=0.2)
        for _, group in outlier_data.groupby('temp_key'):
            ax.plot(group[col_x], group[col_y], group[col_z], color='red', alpha=0.8, linewidth=2)

    # Configuração da estética 3D
    ax.set_title(f'Trajetória 3D: {target_movement}', fontsize=14, pad=20)
    ax.set_xlabel(f'{axis_names[0].upper()} Axis', labelpad=10)
    ax.set_ylabel(f'{axis_names[1].upper()} Axis', labelpad=10)
    ax.set_zlabel(f'{axis_names[2].upper()} Axis', labelpad=10)
    
    # Melhorar a visualização da grelha
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Adicionar a legenda num local que não estorve o gráfico
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.05, 0.95))

    plt.tight_layout()
    plt.show()