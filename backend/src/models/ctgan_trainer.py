"""
Enhanced CTGAN Trainer with RAG-Augmented Generation.

Provides retrieval-augmented training and generation capabilities:
- Conditional generation based on retrieved contexts
- Literature-derived constraints for clinical validity
- Training data augmentation using retrieved information
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json

logger = logging.getLogger(__name__)


@dataclass
class LiteratureConstraint:
    """Constraint derived from medical literature."""
    field: str
    constraint_type: str  # 'range', 'distribution', 'correlation', 'conditional'
    parameters: Dict[str, Any]
    source: str
    confidence: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "constraint_type": self.constraint_type,
            "parameters": self.parameters,
            "source": self.source,
            "confidence": self.confidence
        }


@dataclass
class GenerationContext:
    """Context for conditional generation."""
    conditions: List[str] = field(default_factory=list)
    age_range: Optional[Tuple[int, int]] = None
    gender: Optional[str] = None
    gender_ratio: float = 1.0  # 1.0 = 100% target gender, 0.75 = 75% target gender
    severity: Optional[str] = None
    constraints: List[LiteratureConstraint] = field(default_factory=list)
    retrieved_statistics: Dict[str, Any] = field(default_factory=dict)
    supporting_evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conditions": self.conditions,
            "age_range": self.age_range,
            "gender": self.gender,
            "gender_ratio": self.gender_ratio,
            "severity": self.severity,
            "constraints": [c.to_dict() for c in self.constraints],
            "retrieved_statistics": self.retrieved_statistics,
            "supporting_evidence": self.supporting_evidence
        }


@dataclass
class TrainingAugmentation:
    """Configuration for training data augmentation."""
    augmentation_factor: float = 1.0
    balance_conditions: bool = True
    inject_literature_samples: bool = False
    apply_constraints: bool = True


# Default literature-derived constraints (from major studies)
DEFAULT_LITERATURE_CONSTRAINTS = {
    'age': [
        LiteratureConstraint(
            field='age',
            constraint_type='range',
            parameters={'min': 18, 'max': 100},
            source='Adult patient cohort',
            confidence=1.0
        )
    ],
    'diabetes': [
        LiteratureConstraint(
            field='hba1c',
            constraint_type='conditional',
            parameters={
                'condition': 'has_diabetes == 1',
                'distribution': 'normal',
                'mean': 8.5,
                'std': 2.0,
                'min': 5.7
            },
            source='ADA Standards 2024',
            confidence=0.9
        ),
        LiteratureConstraint(
            field='age',
            constraint_type='conditional',
            parameters={
                'condition': 'has_diabetes == 1',
                'min_typical': 35,
                'mean_shift': 10
            },
            source='UKPDS Study',
            confidence=0.85
        )
    ],
    'hypertension': [
        LiteratureConstraint(
            field='blood_pressure_systolic',
            constraint_type='conditional',
            parameters={
                'condition': 'has_hypertension == 1',
                'min': 130,
                'typical_range': (140, 180)
            },
            source='ACC/AHA Guidelines 2023',
            confidence=0.9
        )
    ],
    'cardiovascular': [
        LiteratureConstraint(
            field='age',
            constraint_type='conditional',
            parameters={
                'condition': 'has_cardiovascular == 1',
                'min_typical': 45,
                'mean_shift': 15
            },
            source='Framingham Heart Study',
            confidence=0.9
        )
    ],
    'sepsis': [
        LiteratureConstraint(
            field='mortality',
            constraint_type='conditional',
            parameters={
                'condition': 'has_sepsis == 1',
                'probability_range': (0.20, 0.35)
            },
            source='Surviving Sepsis Campaign 2021',
            confidence=0.85
        ),
        LiteratureConstraint(
            field='icu_los_days',
            constraint_type='conditional',
            parameters={
                'condition': 'has_sepsis == 1',
                'min': 2,
                'mean': 8,
                'max': 30
            },
            source='Sepsis-3 Epidemiology',
            confidence=0.8
        )
    ],
    'renal': [
        LiteratureConstraint(
            field='creatinine',
            constraint_type='conditional',
            parameters={
                'condition': 'has_renal == 1',
                'min': 1.5,
                'typical_range': (2.0, 8.0)
            },
            source='KDIGO Guidelines',
            confidence=0.9
        )
    ],
    'respiratory': [
        LiteratureConstraint(
            field='oxygen_saturation',
            constraint_type='conditional',
            parameters={
                'condition': 'has_respiratory == 1',
                'max_typical': 94,
                'critical_below': 88
            },
            source='GOLD Report 2024',
            confidence=0.85
        )
    ]
}


class RAGAugmentedCTGANTrainer:
    """
    Enhanced CTGAN trainer with RAG-augmented capabilities.
    
    Usage:
        trainer = RAGAugmentedCTGANTrainer()
        
        # Train with literature constraints
        model = trainer.train(
            training_data,
            apply_constraints=True,
            augmentation_config=TrainingAugmentation(
                augmentation_factor=1.2,
                balance_conditions=True
            )
        )
        
        # Generate with context
        context = GenerationContext(
            conditions=['DIABETES', 'HYPERTENSION'],
            age_range=(65, 85)
        )
        samples = trainer.generate_with_context(model, 100, context)
    """
    
    def __init__(
        self,
        literature_constraints: Optional[Dict] = None,
        rag_system: Optional[Any] = None,
        enhanced_rag: Optional[Any] = None
    ):
        self.literature_constraints = literature_constraints or DEFAULT_LITERATURE_CONSTRAINTS
        self.rag_system = rag_system
        self.enhanced_rag = enhanced_rag
        self._training_history: List[Dict] = []
        self._constraint_cache: Dict[str, List[LiteratureConstraint]] = {}
    
    def train(
        self,
        training_data: pd.DataFrame,
        ctgan_params: Optional[Dict] = None,
        augmentation: Optional[TrainingAugmentation] = None,
        discrete_columns: Optional[List[str]] = None
    ) -> Any:
        """
        Train CTGAN model with optional RAG-based augmentation.
        
        Args:
            training_data: Training DataFrame
            ctgan_params: CTGAN hyperparameters
            augmentation: Augmentation configuration
            discrete_columns: Categorical columns for CTGAN
        """
        try:
            from ctgan import CTGAN
        except ImportError:
            raise ImportError("CTGAN not installed. Install with: pip install ctgan")
        
        augmentation = augmentation or TrainingAugmentation()
        
        # Augment training data if configured
        if augmentation.augmentation_factor > 1.0 or augmentation.balance_conditions:
            training_data = self._augment_training_data(training_data, augmentation)
        
        # Apply literature constraints to training data
        if augmentation.apply_constraints:
            training_data = self._apply_literature_constraints_to_training(training_data)
        
        # Initialize CTGAN
        params = ctgan_params or {
            'epochs': 300,
            'batch_size': 250,
            'generator_dim': (256, 256, 256),
            'discriminator_dim': (256, 256, 256)
        }
        
        ctgan = CTGAN(**params, verbose=True)
        
        # Auto-detect discrete columns if not provided
        if discrete_columns is None:
            discrete_columns = self._detect_discrete_columns(training_data)
        
        logger.info(f"Training CTGAN on {len(training_data)} samples with {len(discrete_columns)} discrete columns")
        
        # Train
        start_time = datetime.now()
        ctgan.fit(training_data, discrete_columns=discrete_columns)
        training_time = datetime.now() - start_time
        
        # Record training history
        self._training_history.append({
            'timestamp': datetime.now().isoformat(),
            'samples': len(training_data),
            'columns': list(training_data.columns),
            'discrete_columns': discrete_columns,
            'training_time': str(training_time),
            'augmentation': {
                'factor': augmentation.augmentation_factor,
                'balanced': augmentation.balance_conditions,
                'constraints_applied': augmentation.apply_constraints
            }
        })
        
        logger.info(f"CTGAN training completed in {training_time}")
        
        return ctgan
    
    def generate_with_context(
        self,
        model: Any,
        num_samples: int,
        context: Optional[GenerationContext] = None,
        post_process: bool = True
    ) -> pd.DataFrame:
        """
        Generate synthetic data with retrieval-augmented context.
        
        Args:
            model: Trained CTGAN model
            num_samples: Number of samples to generate
            context: Generation context with conditions and constraints
            post_process: Apply post-processing for clinical validity
        """
        target = num_samples
        # Start with a higher multiplier for small batches to survive filtering
        base_multiplier = self._calculate_generation_multiplier(context)
        if target <= 50:
            base_multiplier = max(base_multiplier, 3)
        else:
            base_multiplier = max(base_multiplier, 2)
        
        batch_size = min(max(target * base_multiplier, target * 2), 50000)
        max_iters = 5
        collected: List[pd.DataFrame] = []
        total = 0
        
        for _ in range(max_iters):
            logger.info(f"Generating batch of {int(batch_size)} for {target} target (collected={total})")
            batch = model.sample(int(batch_size))
            
            # Apply context-based filtering
            if context:
                batch = self._apply_context_filters(batch, context)
            
            # Apply literature constraints
            if context and context.constraints:
                batch = self._apply_generation_constraints(batch, context.constraints)
            else:
                batch = self._apply_default_constraints(batch, context)
            
            # Post-process for clinical validity
            if post_process:
                batch = self._clinical_post_process(batch, context)
            
            if len(batch) > 0:
                collected.append(batch)
                total += len(batch)
            
            if total >= target:
                break
            
            # Increase batch size modestly if we're still short
            batch_size = min(int(batch_size * 1.5), 50000)
        
        if not collected:
            logger.warning("No valid samples generated; returning empty frame")
            return pd.DataFrame()
        
        synthetic_df = pd.concat(collected, ignore_index=True)
        
        # Pad with replacement if still short to honor the requested count
        if len(synthetic_df) < target:
            need = target - len(synthetic_df)
            pad = synthetic_df.sample(n=min(need, len(synthetic_df)), replace=True)
            synthetic_df = pd.concat([synthetic_df, pad], ignore_index=True)
        
        return synthetic_df.head(target).reset_index(drop=True)
    
    def retrieve_generation_context(
        self,
        query: str,
        conditions: Optional[List[str]] = None
    ) -> GenerationContext:
        """
        Build generation context using RAG retrieval.
        
        Args:
            query: Natural language query describing desired data
            conditions: List of medical conditions
        """
        context = GenerationContext(conditions=conditions or [])
        
        # Use RAG system to retrieve relevant information
        if self.rag_system:
            try:
                rag_result = self.rag_system.extract_relevant_info(query)
                
                # Extract statistics from RAG results
                if rag_result.get('relevant_info'):
                    for info in rag_result['relevant_info']:
                        # Parse any statistical information
                        stats = self._parse_statistics_from_content(info.get('content', ''))
                        context.retrieved_statistics.update(stats)
                        
                        # Add as supporting evidence
                        source = info.get('source', 'Retrieved document')
                        context.supporting_evidence.append(source)
                
            except Exception as e:
                logger.warning(f"Error retrieving context from RAG: {e}")
        
        # Use enhanced RAG for more sources
        if self.enhanced_rag:
            try:
                enhanced_result = self.enhanced_rag.retrieve_and_extract(
                    query,
                    max_results_per_source=10,
                    sources=['pubmed', 'clinical_trials']
                )
                
                # Extract constraints from literature
                if enhanced_result.get('documents'):
                    for doc in enhanced_result['documents']:
                        constraint = self._extract_constraint_from_document(doc)
                        if constraint:
                            context.constraints.append(constraint)
                
            except Exception as e:
                logger.warning(f"Error using enhanced RAG: {e}")
        
        # Add default constraints for conditions
        for condition in context.conditions:
            condition_lower = condition.lower()
            if condition_lower in self.literature_constraints:
                context.constraints.extend(self.literature_constraints[condition_lower])
        
        return context
    
    def add_literature_constraints(
        self,
        condition: str,
        constraints: List[LiteratureConstraint]
    ):
        """Add or update literature constraints for a condition."""
        condition_lower = condition.lower()
        if condition_lower not in self.literature_constraints:
            self.literature_constraints[condition_lower] = []
        
        self.literature_constraints[condition_lower].extend(constraints)
        logger.info(f"Added {len(constraints)} constraints for {condition}")
    
    def _augment_training_data(
        self,
        data: pd.DataFrame,
        config: TrainingAugmentation
    ) -> pd.DataFrame:
        """Augment training data for better model learning."""
        augmented = data.copy()
        
        # Balance conditions if requested
        if config.balance_conditions:
            condition_cols = [c for c in data.columns if c.startswith('has_')]
            
            for col in condition_cols:
                if col in data.columns:
                    positive_samples = data[data[col] == 1]
                    negative_samples = data[data[col] == 0]
                    
                    # Upsample minority class
                    if len(positive_samples) < len(negative_samples) * 0.3:
                        # Condition is rare, upsample
                        target_count = int(len(negative_samples) * 0.3)
                        if len(positive_samples) > 0:
                            additional = positive_samples.sample(
                                n=min(target_count - len(positive_samples), len(positive_samples) * 3),
                                replace=True
                            )
                            augmented = pd.concat([augmented, additional], ignore_index=True)
        
        # Apply augmentation factor
        if config.augmentation_factor > 1.0:
            additional_count = int(len(data) * (config.augmentation_factor - 1.0))
            additional = data.sample(n=additional_count, replace=True)
            
            # Add slight noise to numeric columns
            numeric_cols = additional.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                noise = np.random.normal(0, additional[col].std() * 0.01, len(additional))
                additional[col] = additional[col] + noise
            
            augmented = pd.concat([augmented, additional], ignore_index=True)
        
        logger.info(f"Augmented training data: {len(data)} -> {len(augmented)} samples")
        return augmented
    
    def _apply_literature_constraints_to_training(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply literature constraints to training data for consistency."""
        processed = data.copy()
        
        for condition, constraints in self.literature_constraints.items():
            condition_col = f'has_{condition}'
            
            for constraint in constraints:
                if constraint.field not in processed.columns:
                    continue
                
                if constraint.constraint_type == 'conditional' and condition_col in processed.columns:
                    # Apply conditional constraints
                    mask = processed[condition_col] == 1
                    
                    if 'min' in constraint.parameters:
                        min_val = constraint.parameters['min']
                        processed.loc[mask & (processed[constraint.field] < min_val), constraint.field] = \
                            min_val + np.random.uniform(0, min_val * 0.2, mask.sum())
                    
                    if 'max' in constraint.parameters:
                        max_val = constraint.parameters['max']
                        processed.loc[mask & (processed[constraint.field] > max_val), constraint.field] = max_val
        
        return processed
    
    def _detect_discrete_columns(self, data: pd.DataFrame) -> List[str]:
        """Auto-detect discrete/categorical columns."""
        discrete = []
        
        for col in data.columns:
            if data[col].dtype == 'object':
                discrete.append(col)
            elif data[col].dtype == 'bool':
                discrete.append(col)
            elif col.startswith('has_') or col.startswith('is_'):
                discrete.append(col)
            elif col in ['gender', 'ethnicity', 'insurance', 'admission_type', 
                        'risk_level', 'primary_diagnosis', 'age_group', 'mortality']:
                discrete.append(col)
            elif data[col].nunique() <= 10 and data[col].dtype in ['int64', 'int32']:
                discrete.append(col)
        
        return [c for c in discrete if c in data.columns]
    
    def _calculate_generation_multiplier(self, context: Optional[GenerationContext]) -> int:
        """Calculate how many extra samples to generate for filtering."""
        if not context:
            return 2
        
        multiplier = 2
        
        # Increase for each filter condition
        if context.conditions:
            multiplier += len(context.conditions)
        
        if context.age_range:
            age_span = context.age_range[1] - context.age_range[0]
            if age_span < 20:
                multiplier += 2
        
        if context.gender:
            multiplier += 1
        
        if context.severity == 'critical':
            multiplier += 2
        
        return min(multiplier, 10)
    
    def _apply_context_filters(
        self,
        data: pd.DataFrame,
        context: GenerationContext
    ) -> pd.DataFrame:
        """Apply context-based filters to generated data."""
        filtered = data.copy()
        
        # Filter by conditions
        for condition in context.conditions:
            condition_col = f'has_{condition.lower()}'
            if condition_col in filtered.columns:
                filtered = filtered[filtered[condition_col] == 1]
        
        # Filter by age range
        if context.age_range and 'age' in filtered.columns:
            min_age, max_age = context.age_range
            filtered = filtered[(filtered['age'] >= min_age) & (filtered['age'] <= max_age)]
        
        # Filter by gender (supports soft constraints via gender_ratio)
        if context.gender and 'gender' in filtered.columns:
            gender_mapping = {
                'male': ['male', 'Male', 'MALE', 'M', 'm', 0],
                'female': ['female', 'Female', 'FEMALE', 'F', 'f', 1]
            }
            allowed_values = gender_mapping.get(context.gender.lower(), [context.gender])
            
            # Soft constraint: sample a ratio of target gender vs other
            if context.gender_ratio < 1.0 and len(filtered) > 0:
                target_mask = filtered['gender'].isin(allowed_values)
                target_rows = filtered[target_mask]
                other_rows = filtered[~target_mask]
                
                # Calculate how many of each to keep
                total_target = len(filtered)
                num_target = int(total_target * context.gender_ratio)
                num_other = total_target - num_target
                
                # Sample from each group
                if len(target_rows) >= num_target:
                    sampled_target = target_rows.sample(n=num_target, replace=False)
                else:
                    sampled_target = target_rows  # take all available
                
                if len(other_rows) >= num_other:
                    sampled_other = other_rows.sample(n=num_other, replace=False)
                else:
                    sampled_other = other_rows  # take all available
                
                filtered = pd.concat([sampled_target, sampled_other], ignore_index=True)
                # Shuffle to mix genders
                filtered = filtered.sample(frac=1).reset_index(drop=True)
            else:
                # Hard constraint: 100% target gender
                filtered = filtered[filtered['gender'].isin(allowed_values)]
        
        # Filter by severity
        if context.severity and 'risk_level' in filtered.columns:
            severity_mapping = {
                'critical': ['critical', 'very_high', 4],
                'high': ['high', 3],
                'medium': ['medium', 'moderate', 2],
                'low': ['low', 1]
            }
            if context.severity in severity_mapping:
                allowed = severity_mapping[context.severity]
                filtered = filtered[filtered['risk_level'].isin(allowed)]
        
        return filtered
    
    def _apply_generation_constraints(
        self,
        data: pd.DataFrame,
        constraints: List[LiteratureConstraint]
    ) -> pd.DataFrame:
        """Apply literature-derived constraints to generated data."""
        processed = data.copy()
        
        for constraint in constraints:
            if constraint.field not in processed.columns:
                continue
            
            if constraint.constraint_type == 'range':
                min_val = constraint.parameters.get('min')
                max_val = constraint.parameters.get('max')
                
                if min_val is not None:
                    processed[constraint.field] = processed[constraint.field].clip(lower=min_val)
                if max_val is not None:
                    processed[constraint.field] = processed[constraint.field].clip(upper=max_val)
            
            elif constraint.constraint_type == 'distribution':
                # Adjust to match target distribution
                target_mean = constraint.parameters.get('mean')
                target_std = constraint.parameters.get('std')
                
                if target_mean is not None:
                    current_mean = processed[constraint.field].mean()
                    processed[constraint.field] = processed[constraint.field] + (target_mean - current_mean)
                
                if target_std is not None:
                    current_std = processed[constraint.field].std()
                    if current_std > 0:
                        scale = target_std / current_std
                        mean = processed[constraint.field].mean()
                        processed[constraint.field] = mean + (processed[constraint.field] - mean) * scale
        
        return processed
    
    def _apply_default_constraints(
        self,
        data: pd.DataFrame,
        context: Optional[GenerationContext]
    ) -> pd.DataFrame:
        """Apply default constraints based on detected conditions."""
        processed = data.copy()
        
        if context is None:
            return processed
        
        for condition in context.conditions:
            condition_lower = condition.lower()
            if condition_lower in self.literature_constraints:
                constraints = self.literature_constraints[condition_lower]
                processed = self._apply_generation_constraints(processed, constraints)
        
        return processed
    
    def _clinical_post_process(
        self,
        data: pd.DataFrame,
        context: Optional[GenerationContext]
    ) -> pd.DataFrame:
        """Apply clinical validity post-processing."""
        processed = data.copy()
        
        # Age constraints
        if 'age' in processed.columns:
            processed['age'] = processed['age'].clip(0, 120).round().astype(int)
        
        # ICU LOS <= Hospital LOS
        if 'icu_los_days' in processed.columns and 'hospital_los_days' in processed.columns:
            processed['hospital_los_days'] = np.maximum(
                processed['hospital_los_days'],
                processed['icu_los_days']
            )
        
        # Ensure binary columns are 0/1
        binary_cols = [c for c in processed.columns if c.startswith('has_') or c.startswith('is_')]
        for col in binary_cols:
            processed[col] = (processed[col] > 0.5).astype(int)
        
        # Fix mortality consistency
        if 'mortality' in processed.columns:
            processed['mortality'] = (processed['mortality'] > 0.5).astype(int)
        
        # Apply context-specific post-processing
        if context:
            # Ensure requested conditions are present
            for condition in context.conditions:
                condition_col = f'has_{condition.lower()}'
                if condition_col in processed.columns:
                    # All remaining records should have this condition
                    processed[condition_col] = 1
        
        return processed
    
    def _parse_statistics_from_content(self, content: str) -> Dict[str, Any]:
        """Parse statistical information from retrieved content."""
        import re
        
        stats = {}
        
        # Extract percentages
        pct_pattern = r'(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(?:patients?\s+)?(?:had|with|showed)?\s*(\w+)'
        matches = re.findall(pct_pattern, content, re.IGNORECASE)
        
        for pct, condition in matches:
            try:
                stats[f'{condition.lower()}_rate'] = float(pct) / 100
            except ValueError:
                pass
        
        # Extract means
        mean_pattern = r'mean\s+(\w+)\s+(?:was\s+)?(\d+(?:\.\d+)?)'
        matches = re.findall(mean_pattern, content, re.IGNORECASE)
        
        for field, value in matches:
            try:
                stats[f'{field.lower()}_mean'] = float(value)
            except ValueError:
                pass
        
        return stats
    
    def _extract_constraint_from_document(self, document: Dict) -> Optional[LiteratureConstraint]:
        """Extract a constraint from a retrieved document."""
        content = document.get('content', '') or document.get('abstract', '')
        source = document.get('source', 'Literature')
        
        # Try to find prevalence or statistical information
        import re
        
        # Look for prevalence patterns
        pattern = r'(\w+)\s+(?:prevalence|rate|incidence)\s+(?:was\s+)?(\d+(?:\.\d+)?)\s*%'
        matches = re.findall(pattern, content, re.IGNORECASE)
        
        if matches:
            condition, rate = matches[0]
            return LiteratureConstraint(
                field=f'has_{condition.lower()}',
                constraint_type='distribution',
                parameters={'prevalence': float(rate) / 100},
                source=source,
                confidence=0.7
            )
        
        return None
    
    def save_model(self, model: Any, path: Path, metadata: Optional[Dict] = None):
        """Save trained model with metadata."""
        path = Path(path)
        
        # Save model
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = metadata or {}
        metadata.update({
            'saved_at': datetime.now().isoformat(),
            'training_history': self._training_history,
            'literature_constraints': {
                k: [c.to_dict() for c in v]
                for k, v in self.literature_constraints.items()
            }
        })
        
        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path) -> Tuple[Any, Dict]:
        """Load trained model with metadata."""
        path = Path(path)
        
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        metadata = {}
        metadata_path = path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return model, metadata
