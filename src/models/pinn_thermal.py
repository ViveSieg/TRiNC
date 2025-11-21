"""Gray-box Physics-Informed Neural Network (PiNN) thermal model.

This module implements a gray-box thermal model that learns natural heating patterns
from data and injects physical cooling control terms through explicit equations.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Multi-layer perceptron: maps (Current_Temp, Workload_Vector) -> dT_natural."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3) -> None:
        """Initialize MLP network.
        
        Parameters
        ----------
        input_dim : int
            Workload feature dimension
        hidden_dim : int
            Hidden layer width
        num_layers : int
            Number of hidden layers
        """
        super().__init__()
        layers = []
        in_dim = input_dim + 1  # +1 for current temperature
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
            in_dim = out_dim
        self.net = nn.Sequential(*layers)
    
    def forward(self, current_temp: torch.Tensor, workload: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        current_temp : torch.Tensor
            Current temperature, shape (batch_size, 1)
        workload : torch.Tensor
            Workload vector, shape (batch_size, input_dim)
        
        Returns
        -------
        torch.Tensor
            Natural heating rate dT_natural, shape (batch_size, 1)
        """
        x = torch.cat([current_temp, workload], dim=1)
        return self.net(x)


class GrayBoxThermalModel(nn.Module):
    """Gray-box thermal model combining learned natural heating with physics-based cooling.
    
    The model implements the physics fusion formula:
        dT_total = NN(T_curr, W_curr) - gamma * u
    
    where NN fits natural heating patterns and -gamma * u is the manually injected
    physical cooling term (key to handling data without control signals).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        physics_gamma: float = 0.1,
        Ts: float = 0.1,
        device: str = "cpu",
    ) -> None:
        """Initialize gray-box thermal model.
        
        Parameters
        ----------
        input_dim : int
            Workload feature dimension
        hidden_dim : int
            Network hidden layer width
        physics_gamma : float
            Physical cooling coefficient (fixed or learnable)
        Ts : float
            Sampling time
        device : str
            Computing device ('cpu' or 'cuda')
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.physics_gamma = physics_gamma
        self.Ts = Ts
        self.device = device
        
        self.net = MLP(input_dim, hidden_dim).to(device)
        self.is_trained = False
        
        # Normalization parameters (will be set during training)
        self.input_min: Optional[np.ndarray] = None
        self.input_max: Optional[np.ndarray] = None
        self.output_min: Optional[float] = None
        self.output_max: Optional[float] = None
    
    def step(
        self,
        current_temp: float,
        control_u: float,
        current_workload: np.ndarray,
    ) -> float:
        """Execute one simulation step.
        
        Physics fusion formula:
        1. dT_nat = self.net(T_curr, W_curr)  <-- learned natural heating
        2. dT_cool = -gamma * u                <-- manually injected physical cooling
        3. next_temp = T_curr + (dT_nat + dT_cool) * dt
        
        Parameters
        ----------
        current_temp : float
            Current temperature (normalized)
        control_u : float
            Control signal (normalized to [0, 1])
        current_workload : np.ndarray
            Current workload vector, shape (input_dim,)
        
        Returns
        -------
        float
            Next time step temperature (normalized)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        self.eval()
        with torch.no_grad():
            # Normalize inputs if normalization parameters exist
            if self.input_min is not None and self.input_max is not None:
                workload_normalized = (current_workload - self.input_min) / (
                    self.input_max - self.input_min + 1e-8
                )
            else:
                workload_normalized = current_workload
            
            temp_tensor = torch.tensor([[current_temp]], dtype=torch.float32, device=self.device)
            workload_tensor = torch.tensor(
                [workload_normalized], dtype=torch.float32, device=self.device
            )
            
            # Neural network predicts natural heating rate
            dT_nat = self.net(temp_tensor, workload_tensor).cpu().item()
            
            # Denormalize if needed
            if self.output_min is not None and self.output_max is not None:
                dT_nat = dT_nat * (self.output_max - self.output_min) + self.output_min
            
            # Physics fusion: combine learned heating with physical cooling
            dT_cool = -self.physics_gamma * control_u
            dT_total = dT_nat + dT_cool
            next_temp = current_temp + dT_total * self.Ts
        
        return np.clip(next_temp, 0.0, 1.0)
    
    def reset(self, initial_temperature: float) -> None:
        """Reset model state (stateless model, for interface compatibility only)."""
        pass
    
    def train(
        self,
        train_inputs: np.ndarray,
        train_outputs: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
    ) -> None:
        """Train PiNN model.
        
        Training loss: MSE((T_next - T_curr)/dt - dT_nat)
        Note: Training does not include control signal u, only learns natural heating patterns.
        
        Parameters
        ----------
        train_inputs : np.ndarray
            Training inputs, shape (N, Seq, Feat) - workload features
        train_outputs : np.ndarray
            Training outputs, shape (N, Seq) - temperature sequences (T_max extracted)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        lr : float
            Learning rate
        """
        logger.info(f"Starting PiNN training: {epochs} epochs, batch_size={batch_size}")
        
        # Normalize data using Min-Max normalization
        logger.info("Normalizing input and output data...")
        
        # Normalize inputs (workload features)
        input_flat = train_inputs.reshape(-1, train_inputs.shape[-1])
        self.input_min = np.min(input_flat, axis=0)
        self.input_max = np.max(input_flat, axis=0)
        input_normalized = (input_flat - self.input_min) / (self.input_max - self.input_min + 1e-8)
        train_inputs_normalized = input_normalized.reshape(train_inputs.shape)
        
        # Normalize outputs (temperature)
        output_flat = train_outputs.flatten()
        self.output_min = np.min(output_flat)
        self.output_max = np.max(output_flat)
        train_outputs_normalized = (train_outputs - self.output_min) / (
            self.output_max - self.output_min + 1e-8
        )
        
        # Convert data to (T_curr, workload) -> dT_nat format
        X_list = []
        y_list = []
        
        for seq_idx in range(train_inputs_normalized.shape[0]):
            workload_seq = train_inputs_normalized[seq_idx]  # (Seq, Feat)
            temp_seq = train_outputs_normalized[seq_idx]  # (Seq,)
            
            for t in range(len(temp_seq) - 1):
                T_curr = temp_seq[t]
                T_next = temp_seq[t + 1]
                dT_nat = (T_next - T_curr) / self.Ts
                
                X_list.append((T_curr, workload_seq[t]))
                y_list.append(dT_nat)
        
        X_temps = np.array([x[0] for x in X_list])
        X_workloads = np.array([x[1] for x in X_list])
        y = np.array(y_list)
        
        logger.info(f"Training dataset size: {len(X_list)} samples")
        
        # Convert to PyTorch tensors
        X_temp_tensor = torch.tensor(X_temps, dtype=torch.float32, device=self.device).unsqueeze(1)
        X_workload_tensor = torch.tensor(X_workloads, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # Training setup
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(X_temp_tensor, X_workload_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_temps, batch_workloads, batch_y in dataloader:
                optimizer.zero_grad()
                pred = self.net(batch_temps, batch_workloads)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 20 == 0 or epoch == 0:
                logger.info(f"Training PiNN: Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        self.is_trained = True
        self.eval()
        logger.info("PiNN training completed")
    
    def save(self, path: Path) -> None:
        """Save model weights and normalization parameters.
        
        Parameters
        ----------
        path : Path
            Path to save checkpoint
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'physics_gamma': self.physics_gamma,
            'Ts': self.Ts,
            'input_min': self.input_min,
            'input_max': self.input_max,
            'output_min': self.output_min,
            'output_max': self.output_max,
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "GrayBoxThermalModel":
        """Load model weights and normalization parameters.
        
        Parameters
        ----------
        path : Path
            Path to checkpoint
        device : str
            Computing device
        
        Returns
        -------
        GrayBoxThermalModel
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            physics_gamma=checkpoint['physics_gamma'],
            Ts=checkpoint['Ts'],
            device=device,
        )
        model.net.load_state_dict(checkpoint['net_state_dict'])
        model.is_trained = True
        model.input_min = checkpoint.get('input_min')
        model.input_max = checkpoint.get('input_max')
        model.output_min = checkpoint.get('output_min')
        model.output_max = checkpoint.get('output_max')
        logger.info(f"Model loaded from {path}")
        return model


def load_data(
    data_path: Path, train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load TPU data and split into train/test sets.
    
    Parameters
    ----------
    data_path : Path
        Path to pkl file
    train_ratio : float
        Training set ratio
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (train_inputs, train_outputs, test_inputs, test_outputs)
        - Input: shape (N, Seq, Feat) - multi-dimensional workload metrics
        - Output: shape (N, Seq) - temperature sequences (T_max extracted from thermal maps)
    """
    logger.info(f"Loading data from {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Parse data structure
    if isinstance(data, dict):
        inputs = data.get('inputs', data.get('workload', None))
        outputs = data.get('outputs', data.get('thermal', None))
    elif isinstance(data, (list, tuple)) and len(data) == 2:
        inputs, outputs = data
    else:
        raise ValueError(f"Cannot parse data format: {type(data)}")
    
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    
    # Extract T_max from 2D thermal maps if needed
    if outputs.ndim == 4:
        outputs = np.max(outputs, axis=(2, 3))  # (N, Seq)
    elif outputs.ndim == 3:
        outputs = np.max(outputs, axis=2)  # (N, Seq)
    
    logger.info(f"Data shape: inputs {inputs.shape}, outputs {outputs.shape}")
    
    # Split train/test sets
    N = inputs.shape[0]
    train_size = int(N * train_ratio)
    indices = np.random.RandomState(42).permutation(N)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_inputs = inputs[train_indices]
    train_outputs = outputs[train_indices]
    test_inputs = inputs[test_indices]
    test_outputs = outputs[test_indices]
    
    logger.info(f"Train set: {len(train_indices)} samples, Test set: {len(test_indices)} samples")
    
    return train_inputs, train_outputs, test_inputs, test_outputs


__all__ = ["GrayBoxThermalModel", "load_data"]
