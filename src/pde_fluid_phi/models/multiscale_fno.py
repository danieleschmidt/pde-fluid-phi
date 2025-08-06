"""
Multi-scale Fourier Neural Operator for turbulent flow modeling.

Combines multiple spectral operators at different scales to capture
both large-scale flow structures and small-scale turbulent details.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union
from einops import rearrange

from ..operators.rational_fourier import RationalFourierOperator3D
from ..operators.spectral_layers import SpectralConv3D, MultiScaleOperator
from ..data.spectral_decomposition import SpectralDecomposition
from ..training.losses import MultiScaleLoss


class MultiScaleFNO(nn.Module):
    """
    Multi-scale Fourier Neural Operator.
    
    Processes different spectral scales with specialized operators and
    combines them to model multi-scale turbulent dynamics.
    """
    
    def __init__(
        self,
        scales: List[str] = ['large', 'medium', 'small', 'subgrid'],
        operators_per_scale: Optional[Dict[str, nn.Module]] = None,
        scale_weights: Optional[Dict[str, float]] = None,
        coupling_strength: float = 0.1,
        in_channels: int = 3,
        out_channels: int = 3,
        width: int = 64,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        self.scales = scales
        self.n_scales = len(scales)
        self.coupling_strength = coupling_strength
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        
        # Default scale weights (emphasize smaller scales for turbulence)
        if scale_weights is None:
            scale_weights = {
                'large': 1.0,
                'medium': 2.0, 
                'small': 4.0,
                'subgrid': 8.0
            }
        self.scale_weights = {scale: scale_weights.get(scale, 1.0) for scale in scales}
        
        # Spectral decomposition for scale separation
        self.decomposer = SpectralDecomposition(
            cutoff_wavelengths=[64, 16, 4],  # Reasonable defaults for turbulence
            window='hann'
        )
        
        # Input projections for each scale
        self.input_projections = nn.ModuleDict({
            scale: nn.Linear(in_channels, width) 
            for scale in scales
        })
        
        # Scale-specific operators
        if operators_per_scale is None:
            operators_per_scale = self._create_default_operators()
        self.scale_operators = nn.ModuleDict(operators_per_scale)
        
        # Cross-scale coupling networks
        self.coupling_networks = self._create_coupling_networks()
        
        # Output projections for each scale
        self.output_projections = nn.ModuleDict({
            scale: nn.Linear(width, out_channels)
            for scale in scales
        })
        
        # Final combination network
        self.combination_net = nn.Sequential(
            nn.Conv3d(out_channels * self.n_scales, width, kernel_size=1),
            getattr(nn, activation.upper() if hasattr(nn, activation.upper()) else activation.capitalize())(),
            nn.Conv3d(width, out_channels, kernel_size=1)
        )
        
        # Activation function
        self.activation = getattr(torch.nn.functional, activation)
        
    def _create_default_operators(self) -> Dict[str, nn.Module]:
        """Create default scale-specific operators."""
        operators = {}
        
        for scale in self.scales:
            if scale == 'large':
                # Large scales: fewer modes, focus on global dynamics
                operators[scale] = RationalFourierOperator3D(
                    modes=(16, 16, 16),
                    width=self.width,
                    n_layers=2,
                    in_channels=self.width,
                    out_channels=self.width,
                    rational_order=(2, 2)
                )
            elif scale == 'medium':
                # Medium scales: moderate resolution
                operators[scale] = RationalFourierOperator3D(
                    modes=(32, 32, 32),
                    width=self.width,
                    n_layers=3,
                    in_channels=self.width,
                    out_channels=self.width,
                    rational_order=(3, 3)
                )
            elif scale == 'small':
                # Small scales: high resolution for turbulent structures
                operators[scale] = RationalFourierOperator3D(
                    modes=(64, 64, 64),
                    width=self.width,
                    n_layers=4,
                    in_channels=self.width,
                    out_channels=self.width,
                    rational_order=(4, 4)
                )
            elif scale == 'subgrid':
                # Subgrid scale: simple local model
                operators[scale] = SubgridStressModel(
                    channels=self.width,
                    kernel_size=3
                )
        
        return operators
    
    def _create_coupling_networks(self) -> nn.ModuleDict:
        """Create networks for cross-scale coupling."""
        coupling_nets = nn.ModuleDict()
        
        for i, scale_from in enumerate(self.scales):
            for j, scale_to in enumerate(self.scales):
                if i != j:  # Don't create self-coupling
                    coupling_nets[f"{scale_from}_to_{scale_to}"] = nn.Sequential(
                        nn.Conv3d(self.width, self.width // 4, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv3d(self.width // 4, self.width, kernel_size=1),
                        nn.Tanh()  # Coupling should be bounded
                    )
        
        return coupling_nets
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale FNO.
        
        Args:
            x: Input flow field [batch, channels, height, width, depth]
            
        Returns:
            Predicted flow field [batch, out_channels, height, width, depth]
        """
        batch_size = x.shape[0]
        
        # Step 1: Decompose input into scales
        scale_inputs = self.decomposer.decompose(x)
        
        # Step 2: Process each scale separately
        scale_features = {}
        for scale in self.scales:
            if scale in scale_inputs:
                scale_input = scale_inputs[scale]
            else:
                # Use zero input if scale not available
                scale_input = torch.zeros_like(x)
            
            # Project to feature space
            scale_input_proj = rearrange(scale_input, 'b c h w d -> b h w d c')
            scale_input_proj = self.input_projections[scale](scale_input_proj)
            scale_input_proj = rearrange(scale_input_proj, 'b h w d c -> b c h w d')
            
            # Apply scale-specific operator
            scale_features[scale] = self.scale_operators[scale](scale_input_proj)
        
        # Step 3: Apply cross-scale coupling
        coupled_features = self._apply_coupling(scale_features)
        
        # Step 4: Project each scale to output space
        scale_outputs = []
        for scale in self.scales:
            # Project to output dimension
            scale_output = rearrange(coupled_features[scale], 'b c h w d -> b h w d c')
            scale_output = self.output_projections[scale](scale_output)
            scale_output = rearrange(scale_output, 'b h w d c -> b c h w d')
            
            # Apply scale weight
            scale_output = scale_output * self.scale_weights[scale]
            scale_outputs.append(scale_output)
        
        # Step 5: Combine all scales
        combined = torch.cat(scale_outputs, dim=1)  # Concatenate along channel dimension
        output = self.combination_net(combined)
        
        return output
    
    def _apply_coupling(self, scale_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply cross-scale coupling between different scales."""
        coupled_features = {}
        
        for scale in self.scales:
            coupled_feature = scale_features[scale].clone()
            
            # Add coupling from other scales
            for other_scale in self.scales:
                if other_scale != scale:
                    coupling_key = f"{other_scale}_to_{scale}"
                    if coupling_key in self.coupling_networks:
                        coupling = self.coupling_networks[coupling_key](scale_features[other_scale])
                        coupled_feature = coupled_feature + self.coupling_strength * coupling
            
            coupled_features[scale] = self.activation(coupled_feature)
        
        return coupled_features
    
    def train_multiscale(
        self,
        dataset,
        scale_weights: Optional[Dict[str, float]] = None,
        epochs: int = 100,
        **kwargs
    ):
        """
        Train with multi-scale loss function.
        
        Args:
            dataset: Training dataset
            scale_weights: Weights for different scales in loss
            epochs: Number of training epochs
            **kwargs: Additional training arguments
        """
        if scale_weights is None:
            scale_weights = self.scale_weights
        
        # Create multi-scale loss function
        criterion = MultiScaleLoss(
            scales=self.scales,
            scale_weights=scale_weights,
            decomposer=self.decomposer
        )
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=kwargs.get('lr', 1e-3))
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(dataset):
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(batch, dict):
                    x, y = batch['input'], batch['target']
                else:
                    x, y = batch[0], batch[1]
                
                y_pred = self(x)
                
                # Multi-scale loss
                loss = criterion(y_pred, y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/(batch_idx+1):.6f}")
    
    def get_scale_analysis(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze the flow field at different scales.
        
        Args:
            x: Input flow field [batch, channels, height, width, depth]
            
        Returns:
            Dictionary containing scale analysis results
        """
        # Decompose input
        scale_inputs = self.decomposer.decompose(x)
        
        # Compute energy distribution
        energy_dist = self.decomposer.analyze_energy_cascade(scale_inputs)
        
        # Compute scale interactions
        interactions = self.decomposer.compute_scale_interactions(scale_inputs)
        
        # Get scale info
        scale_info = self.decomposer.get_scale_info()
        
        analysis = {
            'energy_distribution': energy_dist,
            'scale_interactions': interactions,
            'scale_info': scale_info,
            'scale_inputs': scale_inputs
        }
        
        return analysis


class SubgridStressModel(nn.Module):
    """
    Subgrid-scale stress model for small-scale turbulent motions.
    
    Implements a learned subgrid model that can capture effects of
    unresolved scales on the resolved flow field.
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        smagorinsky_constant: float = 0.1,
        learnable_cs: bool = True
    ):
        super().__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        
        # Smagorinsky constant (learnable or fixed)
        if learnable_cs:
            self.cs = nn.Parameter(torch.tensor(smagorinsky_constant))
        else:
            self.register_buffer('cs', torch.tensor(smagorinsky_constant))
        
        # Neural network for enhanced subgrid model
        self.subgrid_net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv3d(channels, channels, kernel_size, padding=kernel_size//2)
        )
        
        # Local averaging for strain rate computation
        self.local_avg = nn.AvgPool3d(kernel_size, stride=1, padding=kernel_size//2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply subgrid-scale model.
        
        Args:
            x: Resolved velocity field [batch, channels, height, width, depth]
            
        Returns:
            Velocity field with subgrid effects [batch, channels, height, width, depth]
        """
        # Compute strain rate tensor (simplified)
        strain_rate = self._compute_strain_rate(x)
        
        # Smagorinsky eddy viscosity
        strain_magnitude = torch.sqrt(2 * torch.sum(strain_rate**2, dim=1, keepdim=True))
        
        # Grid spacing (assumed unit spacing)
        delta = 1.0
        eddy_viscosity = (self.cs * delta)**2 * strain_magnitude
        
        # Apply eddy viscosity (simplified diffusion)
        subgrid_stress = eddy_viscosity * strain_rate
        
        # Neural network enhancement
        neural_correction = self.subgrid_net(x)
        
        # Combine Smagorinsky and neural contributions
        output = x + subgrid_stress + 0.1 * neural_correction
        
        return output
    
    def _compute_strain_rate(self, u: torch.Tensor) -> torch.Tensor:
        """
        Compute strain rate tensor using finite differences.
        
        Args:
            u: Velocity field [batch, 3, height, width, depth]
            
        Returns:
            Strain rate tensor [batch, 6, height, width, depth]
            Components: [S11, S22, S33, S12, S13, S23]
        """
        batch_size = u.shape[0]
        device = u.device
        
        # Compute velocity gradients using finite differences
        du_dx = torch.gradient(u[:, 0], dim=-3)[0]
        du_dy = torch.gradient(u[:, 0], dim=-2)[0]
        du_dz = torch.gradient(u[:, 0], dim=-1)[0]
        
        dv_dx = torch.gradient(u[:, 1], dim=-3)[0]
        dv_dy = torch.gradient(u[:, 1], dim=-2)[0] 
        dv_dz = torch.gradient(u[:, 1], dim=-1)[0]
        
        dw_dx = torch.gradient(u[:, 2], dim=-3)[0]
        dw_dy = torch.gradient(u[:, 2], dim=-2)[0]
        dw_dz = torch.gradient(u[:, 2], dim=-1)[0]
        
        # Strain rate tensor components: S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
        s11 = du_dx  # S11 = du/dx
        s22 = dv_dy  # S22 = dv/dy
        s33 = dw_dz  # S33 = dw/dz
        s12 = 0.5 * (du_dy + dv_dx)  # S12 = 0.5 * (du/dy + dv/dx)
        s13 = 0.5 * (du_dz + dw_dx)  # S13 = 0.5 * (du/dz + dw/dx)
        s23 = 0.5 * (dv_dz + dw_dy)  # S23 = 0.5 * (dv/dz + dw/dy)
        
        # Stack strain rate components
        strain_rate = torch.stack([s11, s22, s33, s12, s13, s23], dim=1)
        
        return strain_rate