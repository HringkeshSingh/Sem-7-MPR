"""
Data processing logic for query parsing and synthetic data generation.
"""

import re
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from fastapi import HTTPException
from src.utils.logging_config import get_logger
from src.utils.common_utils import safe_json_dumps
from src.api.state import app_state

logger = get_logger(__name__)


def parse_natural_language_query(query: str) -> Dict[str, Any]:
    """Parse natural language query to extract filters and conditions."""
    query_lower = query.lower()
    
    filters = {
        'diagnoses': [],
        'diagnosis_logic': 'OR',
        'age_range': None,
        'gender': None,
        'icu_required': None,
        'mortality': None,
        'risk_level': None,
        'complexity': None
    }
    
    # Extract diagnosis conditions
    diagnosis_keywords = {
        'diabetes': ['diabetes', 'diabetic', 'dm', 'glucose'],
        'cardiovascular': ['cardiovascular', 'cardiac', 'heart', 'cvd', 'coronary'],
        'hypertension': ['hypertension', 'hypertensive', 'high blood pressure', 'htn'],
        'renal': ['renal', 'kidney', 'nephritis', 'dialysis', 'ckd'],
        'respiratory': ['respiratory', 'pulmonary', 'copd', 'asthma', 'pneumonia', 'lung', 'conditions'],
        'sepsis': ['sepsis', 'septic', 'infection', 'bacteremia'],
        'neurological': ['neurological', 'stroke', 'seizure', 'brain', 'neuro'],
        'trauma': ['trauma', 'fracture', 'injury', 'accident'],
        'cancer': ['cancer', 'tumor', 'malignant', 'oncology', 'carcinoma'],
    }
    
    for diagnosis, keywords in diagnosis_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            filters['diagnoses'].append(diagnosis.upper())
    
    # Detect AND vs OR logic
    if len(filters['diagnoses']) > 1:
        and_patterns = [' and ', ' with ', ' plus ', ' alongside ', ' combined with ', ' having both ']
        or_patterns = [' or ', ' either ', ' any of ', ' one of ']
        
        has_and = any(p in query_lower for p in and_patterns)
        has_or = any(p in query_lower for p in or_patterns)
        
        if has_and and not has_or:
            filters['diagnosis_logic'] = 'AND'
        elif ' and ' in query_lower and len(filters['diagnoses']) == 2:
            filters['diagnosis_logic'] = 'AND'
    
    # Extract age range
    age_match = re.search(r'(?:aged?|age)\s*(\d+)\s*-\s*(\d+)', query_lower)
    if age_match:
        filters['age_range'] = (int(age_match.group(1)), int(age_match.group(2)))
    elif any(w in query_lower for w in ['elderly', 'old', 'geriatric']):
        filters['age_range'] = (65, 100)
    elif any(w in query_lower for w in ['young', 'adult']):
        filters['age_range'] = (18, 65)
    elif any(w in query_lower for w in ['pediatric', 'child']):
        filters['age_range'] = (0, 18)
    
    # Extract gender
    if re.search(r'\b(female|women)\b', query_lower):
        filters['gender'] = 'female'
    elif re.search(r'\b(male|men)\b', query_lower):
        filters['gender'] = 'male'
    
    # Extract other filters
    if any(w in query_lower for w in ['icu', 'intensive care', 'critical']):
        filters['icu_required'] = True
    
    if any(w in query_lower for w in ['mortality', 'death', 'died']):
        filters['mortality'] = True
    elif any(w in query_lower for w in ['survived', 'survivor']):
        filters['mortality'] = False
    
    if any(w in query_lower for w in ['high risk', 'high-risk', 'severe']):
        filters['risk_level'] = 'high'
    elif any(w in query_lower for w in ['low risk', 'low-risk', 'mild']):
        filters['risk_level'] = 'low'
    
    if any(w in query_lower for w in ['complex', 'multiple', 'comorbid']):
        filters['complexity'] = 'high'
    
    return filters


def apply_filters_to_data(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply parsed filters to the dataset."""
    filtered_df = df.copy()
    
    # Apply diagnosis filters
    if filters['diagnoses']:
        diagnosis_logic = filters.get('diagnosis_logic', 'OR')
        diagnosis_matches = {}
        
        for diagnosis in filters['diagnoses']:
            matches = pd.Series([False] * len(filtered_df))
            
            if 'primary_diagnosis' in filtered_df.columns:
                matches |= (filtered_df['primary_diagnosis'] == diagnosis)
            
            if 'diagnoses' in filtered_df.columns:
                matches |= filtered_df['diagnoses'].str.contains(diagnosis, na=False)
            
            diagnosis_matches[diagnosis] = matches
        
        if diagnosis_logic == 'AND':
            final = pd.Series([True] * len(filtered_df))
            for m in diagnosis_matches.values():
                final &= m
        else:
            final = pd.Series([False] * len(filtered_df))
            for m in diagnosis_matches.values():
                final |= m
        
        filtered_df = filtered_df[final]
    
    # Apply age filter
    if filters['age_range'] and 'age' in filtered_df.columns:
        min_age, max_age = filters['age_range']
        filtered_df = filtered_df[(filtered_df['age'] >= min_age) & (filtered_df['age'] <= max_age)]
    
    # Apply gender filter
    if filters['gender'] and 'gender' in filtered_df.columns:
        target = 'f' if filters['gender'] == 'female' else 'm'
        filtered_df = filtered_df[filtered_df['gender'] == target]
    
    # Apply ICU filter
    if filters['icu_required'] is not None and 'has_icu_stay' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['has_icu_stay'] == (1 if filters['icu_required'] else 0)]
    
    # Apply mortality filter
    if filters['mortality'] is not None and 'mortality' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['mortality'] == (1 if filters['mortality'] else 0)]
    
    # Apply risk level filter
    if filters['risk_level'] and 'risk_level' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['risk_level'] == filters['risk_level']]
    
    # Apply complexity filter
    if filters['complexity'] == 'high' and 'high_complexity' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['high_complexity'] == 1]
    
    return filtered_df


def post_process_synthetic_data(df: pd.DataFrame) -> pd.DataFrame:
    """Post-process synthetic data to ensure clinical validity."""
    processed = df.copy()
    
    # Ensure age is valid
    processed['age'] = processed['age'].clip(18, 100).round().astype(int)
    
    # Ensure LOS is valid
    if 'hospital_los_days' in processed.columns:
        processed['hospital_los_days'] = processed['hospital_los_days'].clip(0, 60).round(1)
    if 'icu_los_days' in processed.columns:
        processed['icu_los_days'] = processed['icu_los_days'].clip(0, 30).round(1)
    
    # Ensure hospital LOS >= ICU LOS
    if 'hospital_los_days' in processed.columns and 'icu_los_days' in processed.columns:
        processed['hospital_los_days'] = np.maximum(processed['hospital_los_days'], processed['icu_los_days'])
    
    # Reconstruct diagnoses if needed
    if 'diagnoses' not in processed.columns:
        diagnosis_map = {
            'DIABETES': ['has_diabetes'],
            'CARDIOVASCULAR': ['has_cardiovascular'],
            'HYPERTENSION': ['has_hypertension'],
            'RENAL': ['has_renal'],
            'RESPIRATORY': ['has_respiratory'],
            'SEPSIS': ['has_sepsis'],
            'NEUROLOGICAL': ['has_neurological'],
        }
        
        diagnoses_list = []
        for _, row in processed.iterrows():
            patient_diagnoses = []
            for diag, cols in diagnosis_map.items():
                for col in cols:
                    if col in processed.columns and row.get(col, 0) > 0.5:
                        patient_diagnoses.append(diag)
                        break
            
            if not patient_diagnoses and 'primary_diagnosis' in row:
                if pd.notna(row['primary_diagnosis']):
                    patient_diagnoses.append(str(row['primary_diagnosis']).upper())
            
            if not patient_diagnoses:
                patient_diagnoses = ['OTHER']
            
            diagnoses_list.append(safe_json_dumps(patient_diagnoses))
        
        processed['diagnoses'] = diagnoses_list
    
    # Ensure valid categorical values
    if 'gender' in processed.columns:
        processed['gender'] = processed['gender'].fillna('unknown')
    
    if 'risk_level' in processed.columns:
        valid = ['low', 'medium', 'high', 'critical']
        processed['risk_level'] = processed['risk_level'].fillna('medium')
        processed['risk_level'] = processed['risk_level'].apply(lambda x: x if x in valid else 'medium')
    
    return processed


def generate_synthetic_data(num_patients: int, filters: Dict[str, Any] = None) -> pd.DataFrame:
    """Generate synthetic data using CTGAN model."""
    if app_state.ctgan_model is None:
        raise HTTPException(status_code=500, detail="CTGAN model not loaded")
    
    logger.info(f"Generating {num_patients} synthetic patients")
    
    synthetic_df = app_state.ctgan_model.sample(num_patients)
    synthetic_df = post_process_synthetic_data(synthetic_df)
    
    if filters and any(filters.values()):
        synthetic_df = apply_filters_to_data(synthetic_df, filters)
        
        max_attempts = 10
        attempt = 0
        
        while len(synthetic_df) < num_patients and attempt < max_attempts:
            attempt += 1
            additional = app_state.ctgan_model.sample(max((num_patients - len(synthetic_df)) * 5, 200))
            additional = post_process_synthetic_data(additional)
            additional = apply_filters_to_data(additional, filters)
            
            if len(additional) > 0:
                synthetic_df = pd.concat([synthetic_df, additional], ignore_index=True)
    
    return synthetic_df.head(num_patients)


def extract_sample_size(query: str) -> int | None:
    """Extract sample size from query, avoiding age ranges."""
    query_lower = query.lower()
    
    patterns = [
        r'\bgenerate\s+(\d+)\b',
        r'\bcreate\s+(\d+)\b',
        r'\b(\d+)\s+patients\b',
        r'\bsample\s+(?:size\s+)?(?:of\s+)?(\d+)\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            return int(match.group(1))
    
    # Avoid age range numbers
    if not re.search(r'(?:aged?|age)\s*\d+\s*-\s*\d+', query_lower):
        match = re.search(r'\b(\d+)\b', query)
        if match:
            return int(match.group(1))
    
    return None
