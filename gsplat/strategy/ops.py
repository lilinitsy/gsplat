import numpy as np
from typing import Callable, Dict, List, Union, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat import quat_scale_to_covar_preci
from gsplat.relocation import compute_relocation
from gsplat.utils import normalized_quat_to_rotmat


@torch.no_grad()
def _multinomial_sample(weights: Tensor, n: int, replacement: bool = True) -> Tensor:
    """Sample from a distribution using torch.multinomial or numpy.random.choice.

    This function adaptively chooses between `torch.multinomial` and `numpy.random.choice`
    based on the number of elements in `weights`. If the number of elements exceeds
    the torch.multinomial limit (2^24), it falls back to using `numpy.random.choice`.

    Args:
        weights (Tensor): A 1D tensor of weights for each element.
        n (int): The number of samples to draw.
        replacement (bool): Whether to sample with replacement. Default is True.

    Returns:
        Tensor: A 1D tensor of sampled indices.
    """
    num_elements = weights.size(0)

    if num_elements <= 2**24:
        # Use torch.multinomial for elements within the limit
        return torch.multinomial(weights, n, replacement=replacement)
    else:
        # Fallback to numpy.random.choice for larger element spaces
        weights = weights / weights.sum()
        weights_np = weights.detach().cpu().numpy()
        sampled_idxs_np = np.random.choice(
            num_elements, size=n, p=weights_np, replace=replacement
        )
        sampled_idxs = torch.from_numpy(sampled_idxs_np)

        # Return the sampled indices on the original device
        return sampled_idxs.to(weights.device)


@torch.no_grad()
def _update_param_with_optimizer(
    param_fn: Callable[[str, Tensor], Tensor],
    optimizer_fn: Callable[[str, Tensor], Tensor],
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    names: Union[List[str], None] = None,
):
    """Update the parameters and the state in the optimizers with defined functions.

    Args:
        param_fn: A function that takes the name of the parameter and the parameter itself,
            and returns the new parameter.
        optimizer_fn: A function that takes the key of the optimizer state and the state value,
            and returns the new state value.
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        names: A list of key names to update. If None, update all. Default: None.
    """
    if names is None:
        # If names is not provided, update all parameters
        names = list(params.keys())

    for name in names:
        param = params[name]
        new_param = param_fn(name, param)
        params[name] = new_param
        if name not in optimizers:
            assert not param.requires_grad, (
                f"Optimizer for {name} is not found, but the parameter is trainable."
                f"Got requires_grad={param.requires_grad}"
            )
            continue
        optimizer = optimizers[name]
        for i in range(len(optimizer.param_groups)):
            param_state = optimizer.state[param]
            del optimizer.state[param]
            for key in param_state.keys():
                if key != "step":
                    v = param_state[key]
                    param_state[key] = optimizer_fn(key, v)
            optimizer.param_groups[i]["params"] = [new_param]
            optimizer.state[new_param] = param_state


def sample_duplicate_blue_noise_positions(
    parent_means: Tensor,  # [N, 3]
    all_means: Tensor,     # [M, 3] all Gaussian positions
    parent_scales: Tensor, # [N]
    min_dist_factor: float = 1.0,
    k_neighbours: int = 10,
    max_attempts: int = 30,
) -> Tensor:
    device = parent_means.device
    n = len(parent_means)
    new_positions = torch.zeros_like(parent_means)
    
    for i in range(n):
        parent_pos = parent_means[i]
        parent_scale = parent_scales[i]
        min_dist = min_dist_factor * parent_scale
        
        # Find k nearest neighbours to parent
        dists = torch.norm(all_means - parent_pos, dim=1)
        (_, neighbour_indices) = torch.topk(dists, k = min(k_neighbours + 1, len(all_means)), largest=False)
        neighbour_indices = neighbour_indices[1:]  # Exclude self
        neighbour_positions = all_means[neighbour_indices]
        
        # Try to sample a valid position
        valid_pos = None
        for attempt in range(max_attempts):
            # Sample random direction and distance
            direction = torch.randn(3, device=device)
            direction = direction / torch.norm(direction)
            distance = torch.rand(1, device=device) * min_dist * 2  # Up to 2x min_dist
            
            candidate = parent_pos + direction * distance
            
            # Check if candidate maintains min_dist from neighbours
            dists_to_neighbours = torch.norm(neighbour_positions - candidate, dim=1)
            if torch.all(dists_to_neighbours >= min_dist):
                valid_pos = candidate
                break
        
        # Fallback to parent position if no valid position found
        if valid_pos is None:
            new_positions[i] = parent_pos
            # print("\nNew position could NOT be found\n") # It's only like 3-4 times per iteration it can't find one
        else:
            new_positions[i] = valid_pos
            #print("\nNew position found\n") # Most of the time it finds a new position

    return new_positions


def sample_split_positions_with_poisson(
    parent_means: Tensor,      # [N, 3]
    parent_scales: Tensor,     # [N, 3]
    parent_rotmats: Tensor,    # [N, 3, 3]
    all_means: Tensor,         # [M, 3]
    min_dist_factor: float = 1.0,
    k_neighbours: int = 20,
    max_attempts: int = 30,
) -> Tuple[Tensor, Tensor]:
    """
    Sample positions for 2 children per parent using dart throwing (Poisson disc sampling)
    
    Returns:
        child1_positions: [N, 3]
        child2_positions: [N, 3]
    """
    device = parent_means.device
    n = len(parent_means)
    child1_positions = torch.zeros_like(parent_means)
    child2_positions = torch.zeros_like(parent_means)

    for i in range(n):
        parent_pos = parent_means[i]
        parent_scale = parent_scales[i].mean()  # Average scale
        rotmat = parent_rotmats[i]  # [3, 3]
        min_dist = min_dist_factor * parent_scale
        
        # Find k nearest neighbours to parent
        dists = torch.norm(all_means - parent_pos, dim=1)
        (_, neighbour_indices) = torch.topk(dists, k=min(k_neighbours + 1, len(all_means)), largest=False)
        neighbour_indices = neighbour_indices[1:]  # Exclude self
        neighbour_positions = all_means[neighbour_indices]
        
        # This will sample child 1  before sampling 2
        # TODO: This can be faster by trying to sample BOTH and accepting when one gets hit
        # and continuing to sample until the second is hit or max_attempts is hit
        child1 = None
        for attempt in range(max_attempts):
            local_direction = torch.randn(3, device=device)
            local_direction = local_direction / torch.norm(local_direction)
            direction = rotmat @ local_direction  # Rotate to world space
            
            distance = torch.rand(1, device=device) * min_dist * 1.5
            candidate1 = parent_pos + direction * distance
            
            dists_to_neighbours = torch.norm(neighbour_positions - candidate1, dim=1)
            if torch.all(dists_to_neighbours >= min_dist):
                child1 = candidate1
                break
        
        if child1 is None:
            # Use original split logic as fallback
            noise = torch.randn(3, device=device)
            child1 = parent_pos + rotmat @ (parent_scales[i] * noise)
        
        child2 = None
        for attempt in range(max_attempts):
            # Random direction
            local_direction = torch.randn(3, device=device)
            local_direction = local_direction / torch.norm(local_direction)
            direction = rotmat @ local_direction
            distance = torch.rand(1, device=device) * min_dist * 1.5
            
            candidate2 = parent_pos + direction * distance
            
            dists_to_neighbours = torch.norm(neighbour_positions - candidate2, dim=1)
            dist_to_child1 = torch.norm(child1 - candidate2)
            
            if torch.all(dists_to_neighbours >= min_dist) and dist_to_child1 >= min_dist:
                child2 = candidate2
                break
        
        if child2 is None:
            # Use original split logic as fallback
            noise = torch.randn(3, device=device)
            child2 = parent_pos - rotmat @ (parent_scales[i] * noise)
        
        child1_positions[i] = child1
        child2_positions[i] = child2
    
    return child1_positions, child2_positions



@torch.no_grad()
def duplicate_with_poisson(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    min_dist_factor: float = 1.0,
    k_neighbours: int = 20,
    max_attempts: int = 30,
):
    """Inplace duplicate the Gaussian with the given mask. Uses poisson disc

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to duplicate the Gaussians.
    """
    device = mask.device
    sel = torch.where(mask)[0]
    n_duplicates = len(sel)
    if n_duplicates == 0:
        return
    parent_means = params["means"][sel]
    parent_scales = torch.exp(params["scales"][sel]).mean(dim=-1)
    all_means = params["means"]
    new_means = sample_duplicate_blue_noise_positions(
        parent_means = parent_means,
        all_means = all_means,
        parent_scales = parent_scales,
        min_dist_factor = min_dist_factor,
        k_neighbours = k_neighbours,
        max_attempts = max_attempts,
    )

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "means":
            return torch.nn.Parameter(
                torch.cat([p, new_means]), 
                requires_grad=p.requires_grad
            )
        else:
            return torch.nn.Parameter(
                torch.cat([p, p[sel]]), 
                requires_grad=p.requires_grad
            )
    
    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat([v, torch.zeros((n_duplicates, *v.shape[1:]), device=device)])
    
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v, v[sel]))



@torch.no_grad()
def duplicate(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
):
    """Inplace duplicate the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to duplicate the Gaussians.
    """
    device = mask.device
    sel = torch.where(mask)[0]

    def param_fn(name: str, p: Tensor) -> Tensor:
        return torch.nn.Parameter(torch.cat([p, p[sel]]), requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v, v[sel]))


@torch.no_grad()
def split_with_poisson(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    min_dist_factor: float = 1.0,
    k_neighbours: int = 20,
    max_attempts: int = 30,
    revised_opacity: bool = False,
):
    """
    Inplace split Gaussians with blue noise spacing.
    
    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers.
        state: Extra running state.
        mask: Boolean mask indicating which Gaussians to split.
        min_dist_factor: Multiplier for minimum distance (relative to parent scale).
        k_neighbours: Number of neighbors to consider for spacing.
        max_attempts: Maximum sampling attempts per child.
        revised_opacity: Whether to use revised opacity formulation.
    """
    device = mask.device
    sel = torch.where(mask)[0] # indicies of gaussians to split
    rest = torch.where(~mask)[0] # indicies of gaussians not to split

    parent_means = params["means"][sel]  # [N, 3]
    parent_scales = torch.exp(params["scales"][sel])  # [N, 3]
    parent_quats = F.normalize(params["quats"][sel], dim=-1)  # [N, 4]
    parent_rotmats = normalized_quat_to_rotmat(parent_quats)  # [N, 3, 3]

    all_means = params["means"] # necessary for placing poisson points

    child1_means, child2_means = sample_split_positions_with_poisson(
        parent_means = parent_means,
        parent_scales = parent_scales,
        parent_rotmats = parent_rotmats,
        all_means = all_means,
        min_dist_factor = min_dist_factor,
        k_neighbours = k_neighbours,
        max_attempts = max_attempts,
    )

    new_means = torch.cat([child1_means, child2_means], dim=0) # combines children
    
    def param_fn(name: str, p: Tensor) -> Tensor:
        repeats = [2] + [1] * (p.dim() - 1)
        
        if name == "means":
            # Use blue noise sampled positions
            p_split = new_means  # [2N, 3]
        elif name == "scales":
            # Smaller scales, repeated for both children
            p_split = torch.log(parent_scales / 1.6).repeat(repeats)  # [2N, 3]
        elif name == "opacities" and revised_opacity:
            new_opacities = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p[sel]))
            p_split = torch.logit(new_opacities).repeat(repeats)  # [2N]
        else:
            # Other parameters (quats, colors) - duplicate from parent
            p_split = p[sel].repeat(repeats)
        
        p_new = torch.cat([p[rest], p_split])
        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
    
    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
        return torch.cat([v[rest], v_split])
    
    # Update parameters and optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    
    # Update extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            state[k] = torch.cat((v[rest], v_new))


    return

@torch.no_grad()
def split(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    revised_opacity: bool = False,
):
    """Inplace split the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to split the Gaussians.
        revised_opacity: Whether to use revised opacity formulation
          from arXiv:2404.06109. Default: False.
    """
    device = mask.device
    sel = torch.where(mask)[0] # indicies of gaussians to split
    rest = torch.where(~mask)[0] # indicies of gaussians not to split

    scales = torch.exp(params["scales"][sel])
    quats = F.normalize(params["quats"][sel], dim=-1)
    rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
    samples = torch.einsum(
        "nij,nj,bnj->bni",
        rotmats,
        scales,
        torch.randn(2, len(scales), 3, device=device),
    )  # [2, N, 3]

    def param_fn(name: str, p: Tensor) -> Tensor:
        repeats = [2] + [1] * (p.dim() - 1)
        if name == "means":
            p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
        elif name == "scales":
            p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
        elif name == "opacities" and revised_opacity:
            new_opacities = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p[sel]))
            p_split = torch.logit(new_opacities).repeat(repeats)  # [2N]
        else:
            p_split = p[sel].repeat(repeats)
        p_new = torch.cat([p[rest], p_split])
        p_new = torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
        return p_new

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
        return torch.cat([v[rest], v_split])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            state[k] = torch.cat((v[rest], v_new))


@torch.no_grad()
def remove(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
):
    """Inplace remove the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to remove the Gaussians.
    """
    sel = torch.where(~mask)[0]

    def param_fn(name: str, p: Tensor) -> Tensor:
        return torch.nn.Parameter(p[sel], requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return v[sel]

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v[sel]


@torch.no_grad()
def reset_opa(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    value: float,
):
    """Inplace reset the opacities to the given post-sigmoid value.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        value: The value to reset the opacities
    """

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            opacities = torch.clamp(p, max=torch.logit(torch.tensor(value)).item())
            return torch.nn.Parameter(opacities, requires_grad=p.requires_grad)
        else:
            raise ValueError(f"Unexpected parameter name: {name}")

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.zeros_like(v)

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(
        param_fn, optimizer_fn, params, optimizers, names=["opacities"]
    )


@torch.no_grad()
def relocate(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,
    binoms: Tensor,
    min_opacity: float = 0.005,
):
    """Inplace relocate some dead Gaussians to the lives ones.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to indicates which Gaussians are dead.
    """
    # support "opacities" with shape [N,] or [N, 1]
    opacities = torch.sigmoid(params["opacities"])

    dead_indices = mask.nonzero(as_tuple=True)[0]
    alive_indices = (~mask).nonzero(as_tuple=True)[0]
    n = len(dead_indices)

    # Sample for new GSs
    eps = torch.finfo(torch.float32).eps
    probs = opacities[alive_indices].flatten()  # ensure its shape is [N,]
    sampled_idxs = _multinomial_sample(probs, n, replacement=True)
    sampled_idxs = alive_indices[sampled_idxs]
    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=torch.exp(params["scales"])[sampled_idxs],
        ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities)
        elif name == "scales":
            p[sampled_idxs] = torch.log(new_scales)
        p[dead_indices] = p[sampled_idxs]
        return torch.nn.Parameter(p, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v[sampled_idxs] = 0
        return v

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            v[sampled_idxs] = 0


@torch.no_grad()
def sample_add(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    n: int,
    binoms: Tensor,
    min_opacity: float = 0.005,
):
    opacities = torch.sigmoid(params["opacities"])

    eps = torch.finfo(torch.float32).eps
    probs = opacities.flatten()
    sampled_idxs = _multinomial_sample(probs, n, replacement=True)
    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=torch.exp(params["scales"])[sampled_idxs],
        ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities)
        elif name == "scales":
            p[sampled_idxs] = torch.log(new_scales)
        p_new = torch.cat([p, p[sampled_idxs]])
        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
        return torch.cat([v, v_new])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v, v_new))


@torch.no_grad()
def inject_noise_to_position(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    scaler: float,
):
    opacities = torch.sigmoid(params["opacities"].flatten())
    scales = torch.exp(params["scales"])
    covars, _ = quat_scale_to_covar_preci(
        params["quats"],
        scales,
        compute_covar=True,
        compute_preci=False,
        triu=False,
    )

    def op_sigmoid(x, k=100, x0=0.995):
        return 1 / (1 + torch.exp(-k * (x - x0)))

    noise = (
        torch.randn_like(params["means"])
        * (op_sigmoid(1 - opacities)).unsqueeze(-1)
        * scaler
    )
    noise = torch.einsum("bij,bj->bi", covars, noise)
    params["means"].add_(noise)
