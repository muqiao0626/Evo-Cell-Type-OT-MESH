import torch
import torch.nn.functional as F
import pandas as pd
from typing import Tuple, Union
from torch import Tensor
from einops import reduce

torch.manual_seed(8)

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.jit.script
def sinkhorn(
    C: Tensor,
    a: Tensor,
    b: Tensor,
    n_sh_iters: int = 5,
    temperature: float = 1,
    u: Union[Tensor, None] = None,
    v: Union[Tensor, None] = None,
    min_clamp: float = 1e-30,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute optimal transport matrix using Sinkhorn algorithm.
    
    Args:
        C: Cost matrix [batch_size, m, n]
        a: Source distribution [batch_size, m]
        b: Target distribution [batch_size, n]
        n_sh_iters: Number of Sinkhorn iterations
        temperature: Temperature parameter for entropic regularization
        u: Initial u vector for warm start
        v: Initial v vector for warm start
        min_clamp: Minimum value for numerical stability
        
    Returns:
        Tuple of (transport matrix, u, v)
    """
    p = -C / temperature
    
    # Clamp to avoid -inf
    log_a = torch.log(a.clamp(min=min_clamp))
    log_b = torch.log(b.clamp(min=min_clamp))

    if u is None:
        u = torch.zeros_like(a)
    if v is None:
        v = torch.zeros_like(b)

    for _ in range(n_sh_iters):
        u = log_a - torch.logsumexp(p + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(p + u.unsqueeze(2), dim=1)

    logT = p + u.unsqueeze(2) + v.unsqueeze(1)
    return logT.exp(), u, v


@torch.enable_grad()
def minimize_entropy_of_sinkhorn(
    C_0: Tensor,
    a: Tensor,
    b: Tensor,
    noise: Union[Tensor, None] = None,
    mesh_lr: float = 1,
    n_mesh_iters: int = 4,
    n_sh_iters: int = 5,
    reuse_u_v: bool = True
) -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
    """
    Apply MESH algorithm to minimize entropy of Sinkhorn transport matrix.
    
    Args:
        C_0: Initial cost matrix
        a: Source distribution
        b: Target distribution
        noise: Optional noise tensor for initialization
        mesh_lr: Learning rate for MESH updates
        n_mesh_iters: Number of MESH iterations
        n_sh_iters: Number of Sinkhorn iterations
        reuse_u_v: Whether to reuse u, v between iterations
        
    Returns:
        Tuple of (modified cost matrix, final u, final v)
    """
    if noise is None:
        noise = torch.randn_like(C_0)

    C_t = C_0 + 0.001 * noise
    C_t.requires_grad_(True)

    u = None
    v = None
    for i in range(n_mesh_iters):
        attn, u, v = sinkhorn(C_t, a, b, u=u, v=v, n_sh_iters=n_sh_iters)

        if not reuse_u_v:
            u = v = None

        entropy = reduce(
            torch.special.entr(attn.clamp(min=1e-20, max=1)), "n a b -> n", "mean"
        ).sum()
        (grad,) = torch.autograd.grad(entropy, C_t, retain_graph=True)
        grad = F.normalize(grad + 1e-20, dim=[1, 2])
        C_t = C_t - mesh_lr * grad

    if not reuse_u_v:
        u = v = None

    return C_t, u, v


def compute_sparse_transport(
    distance_matrix: Union[pd.DataFrame, Tensor],
    mesh_lr: float = 1.0,
    n_mesh_iters: int = 4,
    temperature: float = 1.0,
    n_sh_iters: int = 5
) -> Union[pd.DataFrame, Tensor]:
    """
    Compute sparse optimal transport given a distance matrix using MESH.
    
    Args:
        distance_matrix: Distance/cost matrix
        mesh_lr: Learning rate for MESH updates
        n_mesh_iters: Number of MESH iterations
        temperature: Temperature parameter for Sinkhorn
        n_sh_iters: Number of Sinkhorn iterations
        
    Returns:
        Sparse transport matrix (same type as input)
    """
    # Convert to torch tensor if needed
    if not torch.is_tensor(distance_matrix):
        cost_matrix = torch.tensor(distance_matrix.values, dtype=torch.float32)
        has_index = True
        index = distance_matrix.index
        columns = distance_matrix.columns
    else:
        cost_matrix = distance_matrix
        has_index = False
    
    # Add batch dimension
    cost_matrix = cost_matrix.unsqueeze(0)
    
    # Initialize distributions
    n, m = cost_matrix.shape[1:]
    a = torch.ones(1, n, device=cost_matrix.device) / n
    b = torch.ones(1, m, device=cost_matrix.device) / m
    
    # Apply MESH algorithm
    C_modified, u, v = minimize_entropy_of_sinkhorn(
        cost_matrix, a, b,
        mesh_lr=mesh_lr,
        n_mesh_iters=n_mesh_iters,
        n_sh_iters=n_sh_iters
    )
    
    # Get final transport matrix
    transport_matrix, _, _ = sinkhorn(
        C_modified, a, b,
        n_sh_iters=n_sh_iters,
        temperature=temperature,
        u=u, v=v
    )
    
    # Remove batch dimension
    transport_matrix = transport_matrix.squeeze(0)
    
    # Convert back to DataFrame if input was DataFrame
    if has_index:
        transport_matrix = pd.DataFrame(
            transport_matrix.detach().numpy(),
            index=index,
            columns=columns
        )
    
    return transport_matrix


def compute_regular_transport(
    distance_matrix: Union[pd.DataFrame, Tensor],
    temperature: float = 1.0,
    n_sh_iters: int = 5
) -> Union[pd.DataFrame, Tensor]:
    """
    Compute regular Sinkhorn transport without entropy minimization.
    
    Args:
        distance_matrix: Distance/cost matrix
        temperature: Temperature parameter for Sinkhorn
        n_sh_iters: Number of Sinkhorn iterations
        
    Returns:
        Transport matrix (same type as input)
    """
    # Convert to torch tensor if needed
    if not torch.is_tensor(distance_matrix):
        cost_matrix = torch.tensor(distance_matrix.values, dtype=torch.float32)
        has_index = True
        index = distance_matrix.index
        columns = distance_matrix.columns
    else:
        cost_matrix = distance_matrix
        has_index = False
    
    # Add batch dimension
    cost_matrix = cost_matrix.unsqueeze(0)
    
    # Initialize distributions
    n, m = cost_matrix.shape[1:]
    a = torch.ones(1, n, device=cost_matrix.device) / n
    b = torch.ones(1, m, device=cost_matrix.device) / m
    
    # Apply Sinkhorn
    transport_matrix, _, _ = sinkhorn(
        cost_matrix, a, b,
        n_sh_iters=n_sh_iters,
        temperature=temperature
    )
    
    # Remove batch dimension
    transport_matrix = transport_matrix.squeeze(0)
    
    # Convert back to DataFrame if input was DataFrame
    if has_index:
        transport_matrix = pd.DataFrame(
            transport_matrix.detach().numpy(),
            index=index,
            columns=columns
        )
    
    return transport_matrix

def compute_transport_loss(
    transport_matrix: Union[pd.DataFrame, Tensor],
    distance_matrix: Union[pd.DataFrame, Tensor],
    temperature: float = 1.0
) -> Tensor:
    """
    Compute the transport loss with entropic regularization.
    
    Args:
        transport_matrix: Transport matrix W
        distance_matrix: Distance/cost matrix C
        temperature: Regularization strength (alpha)
        
    Returns:
        Loss value
    """
    # Convert to torch tensor if needed
    if not torch.is_tensor(transport_matrix):
        W = torch.tensor(transport_matrix.values, dtype=torch.float32)
    else:
        W = transport_matrix
    
    if not torch.is_tensor(distance_matrix):
        C = torch.tensor(distance_matrix.values, dtype=torch.float32)
    else:
        C = distance_matrix
    
    # Calculate transport cost
    transport_cost = torch.sum(W * C)
    
    # Calculate entropy regularization
    # We use torch.special.entr which is -x*log(x) for x>0
    # So we need to negate and add W*1
    entropy_reg = -torch.sum(torch.special.entr(W.clamp(min=1e-20))) + torch.sum(W)
    
    # Total loss
    loss = transport_cost + temperature * entropy_reg
    
    return loss