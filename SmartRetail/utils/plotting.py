from pathlib import Path
from typing import List, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

def plot_movement_data(
    clean_data: Union[np.ndarray, pd.DataFrame],
    outlier_data: Union[np.ndarray, pd.DataFrame],
    algorithm_name: str,
    target_movement: str,
    model_representation: str,
    other_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    other_data_legend_label: Optional[str] = None,
    axis_names: List[str] = ['x', 'y', 'z'],
    wrists: List[str] = ['L', 'R'],
    save_fig: bool = False,
    save_path: Optional[Union[str, Path]] = None
):
    """
    Plots movement data for both clean and outlier samples across two wrists (L and R),
    compatible with DTW (np.ndarray) and Feature Extraction (pd.DataFrame) models.

    Parameters:
    - clean_data: np.ndarray (shape: n_samples, T, 6) or pd.DataFrame with columns like 'x_L', 'x_R'.
    - outlier_data: np.ndarray or pd.DataFrame (same format as clean_data).
    - algorithm_name: Name of the algorithm used (e.g., 'DTW', 'DBSCAN').
    - target_movement: The movement type (e.g., 'Park', 'Move', 'Pick').
    - model_representation: Model representation (e.g., 'DTW', 'FeatureBased').
    - other_data: Additional data to plot (same format as clean_data).
    - other_data_legend_label: Legend label for other_data.
    - axis_names: Base names of the axes (default: ['x', 'y', 'z']).
    - wrists: Suffixes identifying the wrists (default: ['L', 'R']).
    - save_fig: Whether to save the figure to disk.
    - save_path: Directory path where the figure will be saved.
    """
    if len(clean_data) == 0 and len(outlier_data) == 0:
        raise ValueError("Ambos 'clean_data' e 'outlier_data' estão vazios. Não há dados para plotar.")

    is_numpy = isinstance(clean_data, np.ndarray) and isinstance(outlier_data, np.ndarray)
    is_df = isinstance(clean_data, pd.DataFrame) and isinstance(outlier_data, pd.DataFrame)

    if not (is_numpy or is_df):
        raise TypeError("'clean_data' e 'outlier_data' devem ser ambos np.ndarray ou ambos pd.DataFrame.")

    # if the Dataset does not have the _L or _R suffix, assume it's a single wrist dataset
    if is_df and f"{axis_names[0]}_{wrists[0]}" not in clean_data.columns:
        active_wrists = [""]
    elif is_numpy and clean_data.shape[-1] == len(axis_names):
        active_wrists = [""]
    else:
        active_wrists = wrists

    n_rows = len(active_wrists)
    n_cols = len(axis_names)
    
    # squeeze=False garante que axs é sempre uma matriz 2D [row, col], mesmo se n_rows == 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    # Configurar elementos da legenda (sem duplicar dentro do loop)
    legend_elements = [
        Line2D([0], [0], color='gray', lw=2, alpha=0.6, label='Clean'),
        Line2D([0], [0], color='red', lw=2, alpha=0.9, label='Outlier'),
    ]
    if other_data is not None:
        label_other = other_data_legend_label if other_data_legend_label else 'Other Data'
        legend_elements.append(Line2D([0], [0], color='blue', lw=2, alpha=0.6, label=label_other))

    wrist_titles = {'L': 'Left Wrist', 'R': 'Right Wrist', '': ''}

    for row_idx, wrist in enumerate(active_wrists):
        for col_idx, axis in enumerate(axis_names):
            ax = axs[row_idx, col_idx]
            
            # Define o nome da coluna (ex: 'x_L') e o índice para numpy (0..2 para L, 3..5 para R)
            col_name = f"{axis}_{wrist}" if wrist else axis
            feature_idx = (row_idx * n_cols) + col_idx

            # --- DTW plot (numpy arrays) ---
            if is_numpy:
                if clean_data.shape[-1] <= feature_idx:
                    raise ValueError(f"O array NumPy tem {clean_data.shape[-1]} features, mas esperava pelo menos {feature_idx + 1}.")
                
                for series in clean_data:
                    ax.plot(series[:, feature_idx], color='gray', alpha=0.3)
                for series in outlier_data:
                    ax.plot(series[:, feature_idx], color='red', alpha=0.8, linewidth=2)

            # --- Feature Extraction Plot ---
            elif is_df:
                if col_name not in clean_data.columns or col_name not in outlier_data.columns:
                    raise KeyError(f"Coluna '{col_name}' não encontrada nos DataFrames.")

                for _, group in clean_data.groupby('temp_key'):
                    ax.plot(group['step'], group[col_name], color='gray', alpha=0.15)
                for _, group in outlier_data.groupby('temp_key'):
                    ax.plot(group['step'], group[col_name], color='red', alpha=0.6, linewidth=1.5)

            # --- Other Data Plot ---
            if other_data is not None:
                if isinstance(other_data, np.ndarray):
                    for series in other_data:
                        ax.plot(series[:, feature_idx], color='blue', alpha=0.3)
                elif isinstance(other_data, pd.DataFrame):
                    for _, group in other_data.groupby('temp_key'):
                        ax.plot(group['step'], group[col_name], color='blue', alpha=0.15)

            title_prefix = f"{wrist_titles.get(wrist, wrist)} - " if wrist else ""
            ax.set_title(f"{title_prefix}{axis.upper()} Axis ({col_name})")
            ax.set_xlabel('Step')
            ax.set_ylabel(f'Coordinate {col_name}')
            ax.grid(True, linestyle=':', alpha=0.6)

            # legend only on the last column of the first row
            if row_idx == 0 and col_idx == n_cols - 1:
                ax.legend(handles=legend_elements, loc='upper right')

    plt.suptitle(
        f'Movement Analysis ({algorithm_name} - {model_representation}): {target_movement} (Clean vs Outliers)',
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta o layout para não sobrepor o suptitle

    if save_fig:
        if save_path is None:
            raise ValueError("O parâmetro 'save_path' deve ser fornecido quando 'save_fig=True'.")
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{algorithm_name}_{target_movement}_{model_representation}_movement_analysis.png"
        plt.savefig(save_dir / file_name, dpi=300)

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