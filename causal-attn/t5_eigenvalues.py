import torch
from transformers import T5ForConditionalGeneration
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import math

def get_matrix_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    # ensure matrix is 2d
    if matrix.dim() < 2:
        return torch.tensor([])
    elif matrix.dim() > 2:
        original_shape = matrix.shape
        matrix = matrix.reshape(original_shape[0], -1)
    
    # make matrix square if needed
    rows, cols = matrix.shape
    if rows != cols:
        size = min(rows, cols)
        matrix = matrix[:size, :size]
    
    try:
        return torch.linalg.eigvals(matrix)
    except Exception as e:
        print(f"Error computing eigenvalues: {e}")
        return torch.tensor([])

def analyze_value_matrices(model_name: str = "t5-small") -> Dict[str, List[Tuple[int, torch.Tensor]]]:
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    eigenvalues_dict = {}
    
    def process_attention_block(attention_block, block_name: str) -> List[Tuple[int, torch.Tensor]]:
        layer_eigenvalues = []
        
        try:
            value_weight = attention_block.v.weight
            num_heads = attention_block.n_heads
            
            if hasattr(attention_block, 'key_value_proj_dim'):
                head_dim = attention_block.key_value_proj_dim
            else:
                head_dim = value_weight.shape[0] // num_heads
            
            try:
                value_heads = value_weight.reshape(num_heads, head_dim, -1)
                for head_idx in range(num_heads):
                    head_matrix = value_heads[head_idx].detach()
                    eigenvals = get_matrix_eigenvalues(head_matrix)
                    layer_eigenvalues.append((head_idx, eigenvals))
            except RuntimeError:
                eigenvals = get_matrix_eigenvalues(value_weight.detach())
                layer_eigenvalues.append((0, eigenvals))
                
        except Exception as e:
            print(f"Error processing {block_name}: {e}")
            
        return layer_eigenvalues
    
    # process encoder
    for encoder_idx, encoder_block in enumerate(model.encoder.block):
        try:
            attention_block = encoder_block.layer[0].SelfAttention
            eigenvalues = process_attention_block(attention_block, f"encoder_layer_{encoder_idx}")
            eigenvalues_dict[f"encoder_layer_{encoder_idx}"] = eigenvalues
        except Exception as e:
            print(f"Error processing encoder layer {encoder_idx}: {e}")
    
    # process decoder
    for decoder_idx, decoder_block in enumerate(model.decoder.block):
        try:
            self_attention = decoder_block.layer[0].SelfAttention
            self_eigenvalues = process_attention_block(self_attention, f"decoder_self_layer_{decoder_idx}")
            eigenvalues_dict[f"decoder_self_layer_{decoder_idx}"] = self_eigenvalues
            
            cross_attention = decoder_block.layer[1].EncDecAttention
            cross_eigenvalues = process_attention_block(cross_attention, f"decoder_cross_layer_{decoder_idx}")
            eigenvalues_dict[f"decoder_cross_layer_{decoder_idx}"] = cross_eigenvalues
        except Exception as e:
            print(f"Error processing decoder layer {decoder_idx}: {e}")
    
    return eigenvalues_dict

def analyze_eigenvalue_properties(eigenvalues_dict: Dict[str, List[Tuple[int, torch.Tensor]]]) -> None:
    for layer_name, layer_eigenvalues in eigenvalues_dict.items():
        print(f"\nAnalyzing {layer_name}:")
        
        for head_idx, eigenvals in layer_eigenvalues:
            if len(eigenvals) == 0:
                print(f"\nHead {head_idx}: No valid eigenvalues computed")
                continue
                
            real_eigenvals = eigenvals.real
            sorted_eigenvals, _ = torch.sort(torch.abs(real_eigenvals), descending=True)
            
            print(f"\nHead {head_idx}:")
            print(f"  Largest eigenvalue (magnitude): {sorted_eigenvals[0]:.4f}")
            print(f"  Smallest eigenvalue (magnitude): {sorted_eigenvals[-1]:.4f}")
            print(f"  Condition number: {sorted_eigenvals[0]/sorted_eigenvals[-1]:.4f}")
            
            imag_ratio = torch.abs(eigenvals.imag).mean() / torch.abs(eigenvals.real).mean()
            print(f"  Ratio of imaginary to real parts: {imag_ratio:.4f}")

def plot_eigenvalue_spectrum(eigenvalues_dict: Dict[str, List[Tuple[int, torch.Tensor]]], 
                           layer_name: str, 
                           head_idx: int,
                           save_path: Optional[str] = None) -> None:
    try:
        eigenvals = eigenvalues_dict[layer_name][head_idx][1]
        
        if len(eigenvals) == 0:
            print(f"No valid eigenvalues to plot for {layer_name}, head {head_idx}")
            return
            
        plt.figure(figsize=(10, 6))
        plt.scatter(eigenvals.real, eigenvals.imag, alpha=0.5)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title(f'Eigenvalue Spectrum for {layer_name}, Head {head_idx}')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error plotting eigenvalue spectrum: {e}")

def plot_layer_head_spectra(eigenvalues_dict: Dict[str, List[Tuple[int, torch.Tensor]]], 
                           layer_name: str,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (15, 15)) -> None:
    plt.style.use('seaborn-v0_8-paper')
    num_heads = len(eigenvalues_dict[layer_name])
    grid_size = int(np.ceil(np.sqrt(num_heads)))
    rows = grid_size
    cols = grid_size
    
    fig = plt.figure(figsize=figsize, dpi=300)
    
    # find global axis limits
    all_eigenvals = []
    for head_idx in range(num_heads):
        eigenvals = eigenvalues_dict[layer_name][head_idx][1]
        all_eigenvals.append(eigenvals)
    
    all_eigenvals = torch.cat(all_eigenvals)
    max_abs_real = torch.max(torch.abs(all_eigenvals.real))
    max_abs_imag = torch.max(torch.abs(all_eigenvals.imag))
    max_abs = max(max_abs_real, max_abs_imag) * 1.1
    
    gs = fig.add_gridspec(rows, cols, hspace=0.4, wspace=0.3)
    
    # plot each head
    for head_idx in range(num_heads):
        ax = fig.add_subplot(gs[head_idx // cols, head_idx % cols])
        eigenvals = eigenvalues_dict[layer_name][head_idx][1]
        
        scatter = ax.scatter(eigenvals.real, eigenvals.imag, 
                           alpha=0.6, s=30,
                           c=torch.abs(eigenvals),
                           cmap='viridis')
        
        ax.set_xlim(-max_abs, max_abs)
        ax.set_ylim(-max_abs, max_abs)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title(f'Head {head_idx}', fontsize=10, pad=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        if head_idx % cols == 0:
            ax.set_ylabel('Imaginary Part', fontsize=9)
        else:
            ax.set_ylabel('')
            
        if head_idx >= num_heads - cols:
            ax.set_xlabel('Real Part', fontsize=9)
        else:
            ax.set_xlabel('')
            
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label('Magnitude', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    plt.suptitle(f'Eigenvalue Spectra for {layer_name}', 
                 fontsize=14, y=0.95)
    
    if save_path:
        plt.savefig(save_path, 
                    bbox_inches='tight',
                    dpi=300,
                    facecolor='white',
                    edgecolor='none')
    else:
        plt.show()
    
    plt.close()

def plot_layer_eigenvalues_grid(eigenvalues_dict: Dict[str, List[Tuple[int, torch.Tensor]]], 
                              layer_name: str,
                              save_plot: bool = False) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle(f'Eigenvalue Spectra for {layer_name}', fontsize=16)
    
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # track min/max values
    real_min = float('inf')
    real_max = float('-inf')
    imag_min = float('inf')
    imag_max = float('-inf')
    
    # get axis limits
    for head_idx, eigenvals in eigenvalues_dict[layer_name]:
        if len(eigenvals) > 0:
            real_min = min(real_min, eigenvals.real.min().item())
            real_max = max(real_max, eigenvals.real.max().item())
            imag_min = min(imag_min, eigenvals.imag.min().item())
            imag_max = max(imag_max, eigenvals.imag.max().item())
    
    padding = 0.1 * max(real_max - real_min, imag_max - imag_min)
    real_min -= padding
    real_max += padding
    imag_min -= padding
    imag_max += padding
    
    axes = axes.flatten()
    for head_idx, eigenvals in eigenvalues_dict[layer_name]:
        row = head_idx // 4
        col = head_idx % 4
        ax = axes[head_idx]
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(color='silver', linestyle=':', linewidth=0.15, zorder=3)
        ax.set_axisbelow(True)
        
        if len(eigenvals) > 0:
            ax.scatter(eigenvals.real.detach().cpu(), 
                      eigenvals.imag.detach().cpu(), 
                      color='crimson',
                      linewidth=0.75,
                      edgecolors='black',
                      s=50)
            
        ax.set_title(f'Head {head_idx}')
        ax.set_xlim(real_min, real_max)
        ax.set_ylim(imag_min, imag_max)
        
        if row != 3:
            ax.set_xticklabels([])
        
        if col != 0:
            ax.set_yticklabels([])
            
        ax.set_aspect('equal', adjustable='box')
    
    fig.text(0.5, 0.02, 'Real Part', ha='center', va='center')
    fig.text(0.02, 0.5, 'Imaginary Part', ha='center', va='center', rotation='vertical')
    
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
    
    if save_plot:
        filename = f"{layer_name}_eigenvalues_grid.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight')
    
    plt.show()
    plt.close()

def t5_get_BV(t5_model, layer_idx=0, head_idx=0, encoder=True):
    if encoder:
        layer = t5_model.encoder.block[layer_idx].layer[0].SelfAttention
    else:
        layer = t5_model.decoder.block[layer_idx].layer[0].SelfAttention
    
    WQ = layer.q.weight.T
    WK = layer.k.weight.T
    WV = layer.v.weight.T
    D = layer.o.weight.T
    
    dk = t5_model.config.d_kv
    dmodel = t5_model.config.d_model
    num_heads = t5_model.config.num_heads
    
    m = head_idx
    WQm = WQ[:, (dk*m):(dk*m + dk)]
    WKm = WK[:, (dk*m):(dk*m + dk)]
    
    WVm = torch.zeros(dmodel, dmodel, device=WV.device)
    WVm[:, (dk*m):(dk*m + dk)] = WV[:, (dk*m):(dk*m + dk)]
    
    b_mtx = torch.matmul(WQm, WKm.T)
    b_mtx = 0.5 * (b_mtx + b_mtx.T) / math.sqrt(dk)
    
    value_mtx = torch.matmul(WVm, D).T
    
    return b_mtx.clone().detach(), value_mtx.clone().detach()

def plot_B_spectra(t5_model, encoder=True):
    num_heads = t5_model.config.num_heads
    num_layers = len(t5_model.encoder.block) if encoder else len(t5_model.decoder.block)
    
    print(f'Note: matrices act on token vectors normalized to ||x||=sqrt({t5_model.config.d_model})')
    
    rows = int(np.ceil(np.sqrt(num_heads)))
    fig, axes = plt.subplots(rows, rows, figsize=(15, 15))
    axes = axes.flatten()
    
    minx, maxx = 0, 0
    betas = []
    dmodel = t5_model.config.d_model
    
    for i in range(num_heads):
        B, _ = t5_get_BV(t5_model, layer_idx=0, head_idx=i, encoder=encoder)
        eigs = torch.linalg.eigvalsh(B)
        eigs = eigs[eigs.abs() > 1e-6]
        
        axes[i].hist(eigs.cpu().numpy(), bins=40, density=True)
        axes[i].set_title(f'head {i}')
        axes[i].set_ylim(0, 8)
        
        minx = min(minx, eigs.min().item())
        maxx = max(maxx, eigs.max().item())
        
        eff_beta = math.sqrt((B.flatten()**2).sum() * dmodel)
        betas.append(eff_beta)
    
    for i in range(num_heads):
        axes[i].set_xlim(minx, maxx)
    
    for i in range(num_heads, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    print('effective betas = ', betas)
    return fig

def plot_V_spectra(t5_model, encoder=True, save_plots=False):
    num_heads = t5_model.config.num_heads
    
    print(f'Note: matrices act on token vectors normalized to ||x||=sqrt({t5_model.config.d_model})')
    
    for i in range(num_heads):
        _, V = t5_get_BV(t5_model, layer_idx=15, head_idx=i, encoder=encoder)
        eigs = torch.linalg.eigvals(V)
        eigs = eigs[eigs.abs() > 1e-4]
        
        print(f'head = {i}, non-zero eigs = {len(eigs)}')
        
        plt.figure(figsize=(10, 10))
        label_size = 16
        plt.rcParams['xtick.labelsize'] = label_size
        plt.rcParams['ytick.labelsize'] = label_size
        
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.grid(color='silver', linestyle=':', linewidth=0.15, zorder=3)
        plt.gca().set_axisbelow(True)
        
        plt.scatter(eigs.real.cpu(), 
                   eigs.imag.cpu(), 
                   color='crimson',
                   linewidth=0.75,
                   edgecolors='black')
        
        plt.title(f'eigenvalues of value matrix for head {i+1}')
        plt.xlim(-2.35, 1.6)
        plt.ylim(-1.6, 1.6)
        plt.gca().set_aspect('equal', adjustable='box')
        
        if save_plots:
            filename = f"t5_eigs_{i+1}.pdf"
            plt.savefig(filename, format='pdf', bbox_inches='tight')
        
        plt.show()
        plt.close()

def extract_t5_value_matrices(model_name: str = "t5-small") -> Dict[str, List[Tuple[int, torch.Tensor]]]:
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    eigenvalues_dict = {}
    
    # process encoder layers
    for layer_idx in range(len(model.encoder.block)):
        print(f"processing layer {layer_idx} of encoder")
        layer_eigenvalues = []
        
        layer = model.encoder.block[layer_idx].layer[0].SelfAttention
        WV = layer.v.weight.T
        D = layer.o.weight.T
        
        dk = model.config.d_kv
        dmodel = model.config.d_model
        num_heads = model.config.num_heads
        
        for head_idx in range(num_heads):
            WVm = torch.zeros(dmodel, dmodel, device=WV.device)
            WVm[:, (dk*head_idx):(dk*head_idx + dk)] = WV[:, (dk*head_idx):(dk*head_idx + dk)]
            
            value_mtx = torch.matmul(WVm, D).T
            eigenvals = torch.linalg.eigvals(value_mtx)
            eigenvals = eigenvals[eigenvals.abs() > 1e-4]
            
            layer_eigenvalues.append((head_idx, eigenvals))
        
        eigenvalues_dict[f"encoder_layer_{layer_idx}"] = layer_eigenvalues
    
    return eigenvalues_dict

def analyze_t5_eigenvalues(model_name: str = "t5-small", save_plots: bool = False) -> None:
    print(f"analyzing eigenvalues for {model_name}...")
    eigenvalues_dict = extract_t5_value_matrices(model_name)
    
    analyze_eigenvalue_properties(eigenvalues_dict)
    
    for layer_name in eigenvalues_dict:
        plot_layer_eigenvalues_grid(eigenvalues_dict, layer_name, save_plots)
            
    return eigenvalues_dict

if __name__ == "__main__":
    model_name = "t5-large"
    model = analyze_t5_eigenvalues(model_name, save_plots=True)