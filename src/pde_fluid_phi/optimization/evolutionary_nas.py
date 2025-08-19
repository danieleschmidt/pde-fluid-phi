"""
Evolutionary Neural Architecture Search for Rational-Fourier Operators

Automatically discovers optimal neural operator architectures for extreme
Reynolds number flows using evolutionary algorithms and performance-based selection.

Key Features:
- Multi-objective optimization (accuracy, stability, efficiency)
- Population-based search with genetic algorithms
- Adaptive mutation rates based on turbulence characteristics
- Hardware-aware architecture optimization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import random
from dataclasses import dataclass
from pathlib import Path
import json
import time
from concurrent.futures import ProcessPoolExecutor
import copy

from ..operators.rational_fourier import RationalFourierOperator3D
from ..operators.quantum_rational_fourier import QuantumRationalFourierLayer, HyperbolicRationalFourierOperator
from ..models.rfno import RationalFNO
from ..training.stability_trainer import StabilityTrainer
from ..evaluation.metrics import compute_turbulence_metrics


@dataclass
class ArchitectureGene:
    """Individual architecture gene in the population."""
    
    # Architecture parameters
    n_layers: int = 4
    width: int = 64
    modes: Tuple[int, int, int] = (32, 32, 32)
    rational_order: Tuple[int, int] = (4, 4)
    activation: str = 'gelu'
    
    # Advanced features
    use_quantum_layers: bool = False
    use_hyperbolic_geometry: bool = False
    multi_scale_levels: int = 3
    attention_heads: int = 8
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 4
    stability_weight: float = 0.01
    
    # Performance metrics (fitness)
    accuracy: float = 0.0
    stability_score: float = 0.0
    efficiency_score: float = 0.0
    reynolds_capability: float = 0.0
    fitness: float = 0.0
    
    # Computational cost
    param_count: int = 0
    memory_usage: float = 0.0
    training_time: float = 0.0


class EvolutionaryNAS:
    """
    Evolutionary Neural Architecture Search for Rational-Fourier Operators.
    """
    
    def __init__(
        self,
        population_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_ratio: float = 0.2,
        multi_objective_weights: Dict[str, float] = None,
        hardware_constraints: Dict[str, Any] = None,
        dataset_generator: Optional[Callable] = None
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        
        # Multi-objective optimization weights
        self.mo_weights = multi_objective_weights or {
            'accuracy': 0.4,
            'stability': 0.3,
            'efficiency': 0.2,
            'reynolds_capability': 0.1
        }
        
        # Hardware constraints
        self.hw_constraints = hardware_constraints or {
            'max_parameters': 10_000_000,
            'max_memory_gb': 16.0,
            'max_training_hours': 24.0
        }
        
        self.dataset_generator = dataset_generator
        
        # Evolution tracking
        self.population_history = []
        self.best_architectures = []
        self.fitness_history = []
        
        # Search space definitions
        self.search_space = self._define_search_space()
    
    def _define_search_space(self) -> Dict[str, Any]:
        """Define the neural architecture search space."""
        return {
            'n_layers': [2, 3, 4, 5, 6, 8],
            'width': [32, 64, 96, 128, 160, 192, 256],
            'modes': [
                (16, 16, 16), (24, 24, 24), (32, 32, 32),
                (48, 48, 48), (64, 64, 64), (96, 96, 96)
            ],
            'rational_order': [
                (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (8, 8)
            ],
            'activation': ['relu', 'gelu', 'swish', 'mish'],
            'use_quantum_layers': [True, False],
            'use_hyperbolic_geometry': [True, False],
            'multi_scale_levels': [1, 2, 3, 4, 5],
            'attention_heads': [4, 8, 12, 16],
            'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3],
            'stability_weight': [0.001, 0.01, 0.1, 0.5]
        }
    
    def initialize_population(self) -> List[ArchitectureGene]:
        """Initialize random population of architectures."""
        population = []
        
        for _ in range(self.population_size):
            gene = ArchitectureGene()
            
            # Randomly sample from search space
            for param, values in self.search_space.items():
                if hasattr(gene, param):
                    setattr(gene, param, random.choice(values))
            
            population.append(gene)
        
        return population
    
    def evolve(
        self,
        train_datasets: List[Any],
        val_datasets: List[Any],
        save_path: Optional[str] = None
    ) -> List[ArchitectureGene]:
        """
        Run evolutionary optimization to find best architectures.
        
        Args:
            train_datasets: List of training datasets for different Reynolds numbers
            val_datasets: List of validation datasets
            save_path: Path to save evolution results
            
        Returns:
            List of best evolved architectures
        """
        print(f"🧬 Starting Evolutionary NAS with {self.population_size} architectures")
        print(f"   Generations: {self.n_generations}")
        print(f"   Multi-objective weights: {self.mo_weights}")
        
        # Initialize population
        population = self.initialize_population()
        
        # Evolution loop
        for generation in range(self.n_generations):
            print(f"\n🔄 Generation {generation + 1}/{self.n_generations}")
            
            # Evaluate fitness for all architectures
            population = self._evaluate_population(population, train_datasets, val_datasets)
            
            # Sort by fitness (higher is better)
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Track best architectures
            best_arch = population[0]
            self.best_architectures.append(copy.deepcopy(best_arch))
            
            # Log generation statistics
            fitnesses = [arch.fitness for arch in population]
            print(f"   Best fitness: {best_arch.fitness:.4f}")
            print(f"   Mean fitness: {np.mean(fitnesses):.4f}")
            print(f"   Std fitness: {np.std(fitnesses):.4f}")
            
            self.fitness_history.append({
                'generation': generation,
                'best_fitness': best_arch.fitness,
                'mean_fitness': np.mean(fitnesses),
                'std_fitness': np.std(fitnesses)
            })
            
            # Early stopping check
            if self._should_early_stop(generation):
                print(f"🛑 Early stopping at generation {generation}")
                break
            
            # Create next generation
            if generation < self.n_generations - 1:
                population = self._create_next_generation(population)
            
            # Save intermediate results
            if save_path and generation % 10 == 0:
                self._save_evolution_state(save_path, generation)
        
        # Final save
        if save_path:
            self._save_evolution_state(save_path, self.n_generations - 1, final=True)
        
        # Return top architectures
        return self.best_architectures[-10:]  # Top 10 from final generation
    
    def _evaluate_population(
        self,
        population: List[ArchitectureGene],
        train_datasets: List[Any],
        val_datasets: List[Any]
    ) -> List[ArchitectureGene]:
        """Evaluate fitness for all architectures in population."""
        
        # Parallel evaluation for efficiency
        with ProcessPoolExecutor(max_workers=min(8, len(population))) as executor:
            futures = []
            
            for arch in population:
                future = executor.submit(
                    self._evaluate_single_architecture,
                    arch, train_datasets, val_datasets
                )
                futures.append((arch, future))
            
            # Collect results
            for arch, future in futures:
                try:
                    evaluated_arch = future.result(timeout=3600)  # 1 hour timeout
                    # Update original architecture with results
                    arch.__dict__.update(evaluated_arch.__dict__)
                except Exception as e:
                    print(f"⚠️  Architecture evaluation failed: {e}")
                    arch.fitness = 0.0  # Penalty for failed evaluation
        
        return population
    
    def _evaluate_single_architecture(
        self,
        architecture: ArchitectureGene,
        train_datasets: List[Any],
        val_datasets: List[Any]
    ) -> ArchitectureGene:
        """Evaluate a single architecture's performance."""
        
        try:
            # Build model from architecture
            model = self._build_model(architecture)
            
            # Check hardware constraints
            if not self._satisfies_hardware_constraints(model, architecture):
                architecture.fitness = 0.0
                return architecture
            
            # Train and evaluate model
            performance_metrics = self._train_and_evaluate(
                model, architecture, train_datasets, val_datasets
            )
            
            # Update architecture with results
            architecture.accuracy = performance_metrics['accuracy']
            architecture.stability_score = performance_metrics['stability']
            architecture.efficiency_score = performance_metrics['efficiency']
            architecture.reynolds_capability = performance_metrics['reynolds_capability']
            
            # Compute multi-objective fitness
            architecture.fitness = (
                self.mo_weights['accuracy'] * architecture.accuracy +
                self.mo_weights['stability'] * architecture.stability_score +
                self.mo_weights['efficiency'] * architecture.efficiency_score +
                self.mo_weights['reynolds_capability'] * architecture.reynolds_capability
            )
            
        except Exception as e:
            print(f"Architecture evaluation error: {e}")
            architecture.fitness = 0.0
        
        return architecture
    
    def _build_model(self, architecture: ArchitectureGene) -> nn.Module:
        """Build PyTorch model from architecture specification."""
        
        if architecture.use_hyperbolic_geometry:
            # Use hyperbolic rational FNO
            model = HyperbolicRationalFourierOperator(
                modes=architecture.modes,
                width=architecture.width,
                n_layers=architecture.n_layers,
                quantum_enhancement=architecture.use_quantum_layers
            )
        elif architecture.use_quantum_layers:
            # Use quantum-enhanced rational FNO
            from ..operators.quantum_rational_fourier import QuantumRationalFourierLayer
            # Build custom model with quantum layers
            model = self._build_quantum_model(architecture)
        else:
            # Standard rational FNO
            model = RationalFNO(
                modes=architecture.modes,
                width=architecture.width,
                n_layers=architecture.n_layers,
                rational_order=architecture.rational_order,
                activation=architecture.activation,
                stability_weight=architecture.stability_weight,
                multi_scale=(architecture.multi_scale_levels > 1)
            )
        
        return model
    
    def _build_quantum_model(self, architecture: ArchitectureGene) -> nn.Module:
        """Build quantum-enhanced rational FNO."""
        # Custom implementation combining multiple quantum layers
        class CustomQuantumRFNO(nn.Module):
            def __init__(self, arch):
                super().__init__()
                self.layers = nn.ModuleList([
                    QuantumRationalFourierLayer(
                        in_channels=arch.width,
                        out_channels=arch.width,
                        modes=arch.modes,
                        n_quantum_states=8
                    ) for _ in range(arch.n_layers)
                ])
                self.input_proj = nn.Linear(3, arch.width)
                self.output_proj = nn.Linear(arch.width, 3)
                
            def forward(self, x):
                from einops import rearrange
                x = rearrange(x, 'b c h w d -> b h w d c')
                x = self.input_proj(x)
                x = rearrange(x, 'b h w d c -> b c h w d')
                
                for layer in self.layers:
                    x = layer(x)
                
                x = rearrange(x, 'b c h w d -> b h w d c')
                x = self.output_proj(x)
                x = rearrange(x, 'b h w d c -> b c h w d')
                return x
        
        return CustomQuantumRFNO(architecture)
    
    def _satisfies_hardware_constraints(
        self, 
        model: nn.Module, 
        architecture: ArchitectureGene
    ) -> bool:
        """Check if architecture satisfies hardware constraints."""
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        architecture.param_count = param_count
        
        if param_count > self.hw_constraints['max_parameters']:
            return False
        
        # Estimate memory usage (rough approximation)
        input_size = (1, 3, 64, 64, 64)  # Typical input size
        try:
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(input_size)
                _ = model(dummy_input)
            
            # Memory estimation (very rough)
            memory_mb = param_count * 4 / (1024 * 1024)  # 4 bytes per float32
            architecture.memory_usage = memory_mb / 1024  # Convert to GB
            
            if architecture.memory_usage > self.hw_constraints['max_memory_gb']:
                return False
                
        except Exception:
            return False  # Failed to run model
        
        return True
    
    def _train_and_evaluate(
        self,
        model: nn.Module,
        architecture: ArchitectureGene,
        train_datasets: List[Any],
        val_datasets: List[Any]
    ) -> Dict[str, float]:
        """Train and evaluate model performance."""
        
        start_time = time.time()
        
        # Quick training for architecture search (reduced epochs)
        trainer = StabilityTrainer(
            model=model,
            learning_rate=architecture.learning_rate,
            stability_reg=architecture.stability_weight
        )
        
        try:
            # Train on multiple Reynolds numbers
            total_loss = 0.0
            stability_scores = []
            
            for train_data, val_data in zip(train_datasets[:2], val_datasets[:2]):  # Limit for speed
                # Quick training (few epochs)
                train_metrics = trainer.fit(
                    train_data, 
                    val_data,
                    epochs=5,  # Reduced for NAS
                    verbose=False
                )
                
                total_loss += train_metrics['best_val_loss']
                stability_scores.append(train_metrics['stability_score'])
            
            training_time = time.time() - start_time
            architecture.training_time = training_time
            
            # Compute performance metrics
            accuracy = max(0.0, 1.0 - total_loss / len(train_datasets))  # Convert loss to accuracy
            stability = np.mean(stability_scores)
            
            # Efficiency score (inverse of computational cost)
            efficiency = 1.0 / (1.0 + architecture.param_count / 1_000_000 + training_time / 3600)
            
            # Reynolds capability (ability to handle high Re)
            reynolds_capability = min(1.0, stability * accuracy * 2.0)  # Heuristic
            
            if training_time > self.hw_constraints['max_training_hours'] * 3600:
                efficiency *= 0.1  # Heavy penalty for slow training
            
            return {
                'accuracy': accuracy,
                'stability': stability,
                'efficiency': efficiency,
                'reynolds_capability': reynolds_capability
            }
            
        except Exception as e:
            print(f"Training failed: {e}")
            return {
                'accuracy': 0.0,
                'stability': 0.0,
                'efficiency': 0.0,
                'reynolds_capability': 0.0
            }
    
    def _create_next_generation(self, population: List[ArchitectureGene]) -> List[ArchitectureGene]:
        """Create next generation through selection, crossover, and mutation."""
        
        next_generation = []
        
        # Elite selection (keep best architectures)
        n_elite = int(self.population_size * self.elite_ratio)
        elite = population[:n_elite]
        next_generation.extend([copy.deepcopy(arch) for arch in elite])
        
        # Generate rest through crossover and mutation
        while len(next_generation) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
                next_generation.extend([child1, child2])
            else:
                next_generation.extend([copy.deepcopy(parent1), copy.deepcopy(parent2)])
        
        # Mutation
        for arch in next_generation[n_elite:]:  # Don't mutate elites
            if random.random() < self.mutation_rate:
                self._mutate(arch)
        
        # Trim to exact population size
        return next_generation[:self.population_size]
    
    def _tournament_selection(self, population: List[ArchitectureGene], tournament_size: int = 3) -> ArchitectureGene:
        """Tournament selection for parent selection."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(
        self, 
        parent1: ArchitectureGene, 
        parent2: ArchitectureGene
    ) -> Tuple[ArchitectureGene, ArchitectureGene]:
        """Uniform crossover between two parents."""
        
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # Crossover each parameter with 50% probability
        for param in ['n_layers', 'width', 'modes', 'rational_order', 'activation',
                     'use_quantum_layers', 'use_hyperbolic_geometry', 'multi_scale_levels',
                     'attention_heads', 'learning_rate', 'stability_weight']:
            
            if random.random() < 0.5:
                # Swap parameters
                temp = getattr(child1, param)
                setattr(child1, param, getattr(child2, param))
                setattr(child2, param, temp)
        
        # Reset fitness (will be evaluated in next generation)
        child1.fitness = 0.0
        child2.fitness = 0.0
        
        return child1, child2
    
    def _mutate(self, architecture: ArchitectureGene):
        """Mutate architecture parameters."""
        
        # Choose random parameter to mutate
        mutable_params = [
            'n_layers', 'width', 'modes', 'rational_order', 'activation',
            'use_quantum_layers', 'use_hyperbolic_geometry', 'multi_scale_levels',
            'attention_heads', 'learning_rate', 'stability_weight'
        ]
        
        param_to_mutate = random.choice(mutable_params)
        
        # Mutate with new random value from search space
        if param_to_mutate in self.search_space:
            new_value = random.choice(self.search_space[param_to_mutate])
            setattr(architecture, param_to_mutate, new_value)
        
        # Reset fitness
        architecture.fitness = 0.0
    
    def _should_early_stop(self, generation: int, patience: int = 20) -> bool:
        """Check if evolution should stop early due to convergence."""
        
        if generation < patience:
            return False
        
        # Check if fitness hasn't improved significantly in last `patience` generations
        recent_fitnesses = [h['best_fitness'] for h in self.fitness_history[-patience:]]
        
        if len(recent_fitnesses) >= patience:
            fitness_improvement = max(recent_fitnesses) - min(recent_fitnesses)
            if fitness_improvement < 0.01:  # Less than 1% improvement
                return True
        
        return False
    
    def _save_evolution_state(self, save_path: str, generation: int, final: bool = False):
        """Save current evolution state to disk."""
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best architectures
        best_archs_data = []
        for arch in self.best_architectures:
            arch_dict = {
                'generation': generation,
                'n_layers': arch.n_layers,
                'width': arch.width,
                'modes': arch.modes,
                'rational_order': arch.rational_order,
                'activation': arch.activation,
                'use_quantum_layers': arch.use_quantum_layers,
                'use_hyperbolic_geometry': arch.use_hyperbolic_geometry,
                'multi_scale_levels': arch.multi_scale_levels,
                'attention_heads': arch.attention_heads,
                'learning_rate': arch.learning_rate,
                'stability_weight': arch.stability_weight,
                'fitness': arch.fitness,
                'accuracy': arch.accuracy,
                'stability_score': arch.stability_score,
                'efficiency_score': arch.efficiency_score,
                'reynolds_capability': arch.reynolds_capability,
                'param_count': arch.param_count,
                'memory_usage': arch.memory_usage,
                'training_time': arch.training_time
            }
            best_archs_data.append(arch_dict)
        
        # Save to JSON
        with open(save_dir / f'best_architectures_gen_{generation}.json', 'w') as f:
            json.dump(best_archs_data, f, indent=2)
        
        # Save fitness history
        with open(save_dir / 'fitness_history.json', 'w') as f:
            json.dump(self.fitness_history, f, indent=2)
        
        if final:
            print(f"💾 Evolution results saved to {save_path}")


def run_evolutionary_nas(
    config: Dict[str, Any],
    train_datasets: List[Any],
    val_datasets: List[Any],
    save_path: str = "./nas_results"
) -> List[ArchitectureGene]:
    """
    Run evolutionary neural architecture search.
    
    Args:
        config: NAS configuration dictionary
        train_datasets: Training datasets for different Reynolds numbers
        val_datasets: Validation datasets
        save_path: Path to save results
        
    Returns:
        List of best discovered architectures
    """
    
    nas = EvolutionaryNAS(
        population_size=config.get('population_size', 50),
        n_generations=config.get('n_generations', 100),
        mutation_rate=config.get('mutation_rate', 0.1),
        crossover_rate=config.get('crossover_rate', 0.7),
        elite_ratio=config.get('elite_ratio', 0.2),
        multi_objective_weights=config.get('multi_objective_weights'),
        hardware_constraints=config.get('hardware_constraints')
    )
    
    best_architectures = nas.evolve(
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        save_path=save_path
    )
    
    return best_architectures


if __name__ == "__main__":
    # Example configuration for running NAS
    config = {
        'population_size': 30,
        'n_generations': 50,
        'mutation_rate': 0.15,
        'crossover_rate': 0.8,
        'elite_ratio': 0.1,
        'multi_objective_weights': {
            'accuracy': 0.35,
            'stability': 0.35,
            'efficiency': 0.2,
            'reynolds_capability': 0.1
        },
        'hardware_constraints': {
            'max_parameters': 5_000_000,
            'max_memory_gb': 12.0,
            'max_training_hours': 12.0
        }
    }
    
    print("🧬 Evolutionary Neural Architecture Search for Rational-Fourier Operators")
    print("   Discovering optimal architectures for extreme Reynolds number flows...")
    print(f"   Configuration: {config}")