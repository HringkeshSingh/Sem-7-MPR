"""
ClinicalTrials.gov API client for retrieving clinical trial information.

Uses the ClinicalTrials.gov API v2 to search and retrieve clinical trial data.
"""

import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .base_client import BaseDataSourceClient, RetrievedDocument

logger = logging.getLogger(__name__)


class ClinicalTrialsClient(BaseDataSourceClient):
    """
    Client for ClinicalTrials.gov API.
    
    Retrieves clinical trial data including:
    - Trial descriptions
    - Eligibility criteria
    - Outcome measures
    - Study design
    """
    
    BASE_URL = "https://clinicaltrials.gov/api/v2"
    
    def __init__(self, rate_limit: float = 0.5):
        """Initialize the ClinicalTrials.gov client."""
        super().__init__(source_name="clinical_trials", rate_limit=rate_limit)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HealthcareDataGenerator/1.0'
        })
    
    def _check_availability(self) -> bool:
        """Check if ClinicalTrials.gov API is accessible."""
        try:
            response = self.session.get(
                f"{self.BASE_URL}/studies",
                params={"pageSize": 1},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"ClinicalTrials.gov API not available: {e}")
            return False
    
    def search(self, query: str, max_results: int = 20) -> List[RetrievedDocument]:
        """
        Search ClinicalTrials.gov for clinical trials.
        
        Args:
            query: Search query (conditions, interventions, etc.)
            max_results: Maximum number of results
            
        Returns:
            List of RetrievedDocument objects
        """
        self._rate_limit_wait()
        
        try:
            params = {
                "query.term": query,
                "pageSize": min(max_results, 100),
                "format": "json",
                "fields": "NCTId,BriefTitle,OfficialTitle,BriefSummary,DetailedDescription,"
                         "Condition,OverallStatus,Phase,StudyType,EnrollmentCount,"
                         "EligibilityCriteria,StartDate,CompletionDate,LeadSponsorName"
            }
            
            response = self.session.get(
                f"{self.BASE_URL}/studies",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            studies = data.get("studies", [])
            
            documents = []
            for i, study in enumerate(studies):
                doc = self._parse_study(study, i, len(studies))
                if doc:
                    documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} clinical trials for query: {query}")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching ClinicalTrials.gov: {e}")
            return []
    
    def _parse_study(self, study: Dict[str, Any], index: int, total: int) -> Optional[RetrievedDocument]:
        """Parse a clinical trial study into RetrievedDocument format."""
        try:
            protocol = study.get("protocolSection", {})
            id_module = protocol.get("identificationModule", {})
            desc_module = protocol.get("descriptionModule", {})
            status_module = protocol.get("statusModule", {})
            design_module = protocol.get("designModule", {})
            eligibility_module = protocol.get("eligibilityModule", {})
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            
            nct_id = id_module.get("nctId", "")
            title = id_module.get("briefTitle") or id_module.get("officialTitle", "No title")
            
            # Build content from multiple fields
            content_parts = []
            
            if desc_module.get("briefSummary"):
                content_parts.append(f"Summary: {desc_module['briefSummary']}")
            
            if desc_module.get("detailedDescription"):
                content_parts.append(f"Description: {desc_module['detailedDescription']}")
            
            conditions = protocol.get("conditionsModule", {}).get("conditions", [])
            if conditions:
                content_parts.append(f"Conditions: {', '.join(conditions)}")
            
            if eligibility_module.get("eligibilityCriteria"):
                content_parts.append(f"Eligibility: {eligibility_module['eligibilityCriteria']}")
            
            content = "\n\n".join(content_parts) or "No description available"
            
            # Get start date
            start_date = None
            if status_module.get("startDateStruct"):
                date_str = status_module["startDateStruct"].get("date")
                if date_str:
                    try:
                        start_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        try:
                            start_date = datetime.strptime(date_str, "%Y-%m")
                        except ValueError:
                            pass
            
            # Get sponsor
            sponsor = None
            if sponsor_module.get("leadSponsor"):
                sponsor = sponsor_module["leadSponsor"].get("name")
            
            metadata = {
                "nct_id": nct_id,
                "status": status_module.get("overallStatus", "Unknown"),
                "phase": design_module.get("phases", ["Not Applicable"])[0] if design_module.get("phases") else "Not Applicable",
                "study_type": design_module.get("studyType", "Unknown"),
                "enrollment": design_module.get("enrollmentInfo", {}).get("count"),
                "conditions": conditions,
                "sponsor": sponsor
            }
            
            return RetrievedDocument(
                source="clinical_trials",
                source_id=nct_id,
                title=title,
                content=content[:5000],  # Limit content length
                url=f"https://clinicaltrials.gov/study/{nct_id}",
                authors=[sponsor] if sponsor else None,
                publication_date=start_date,
                metadata=metadata,
                relevance_score=1.0 - (index / max(total, 1))
            )
            
        except Exception as e:
            logger.warning(f"Error parsing clinical trial: {e}")
            return None
    
    def fetch_by_id(self, nct_id: str) -> Optional[RetrievedDocument]:
        """
        Fetch a specific clinical trial by NCT ID.
        
        Args:
            nct_id: ClinicalTrials.gov NCT ID (e.g., NCT12345678)
            
        Returns:
            RetrievedDocument or None
        """
        self._rate_limit_wait()
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/studies/{nct_id}",
                params={"format": "json"},
                timeout=30
            )
            response.raise_for_status()
            
            study = response.json()
            return self._parse_study(study, 0, 1)
            
        except Exception as e:
            logger.error(f"Error fetching clinical trial {nct_id}: {e}")
            return None
    
    def build_healthcare_query(
        self,
        conditions: Optional[List[str]] = None,
        demographics: Optional[Dict[str, Any]] = None,
        keywords: Optional[List[str]] = None
    ) -> str:
        """
        Build a query optimized for ClinicalTrials.gov.
        
        Args:
            conditions: Medical conditions
            demographics: Patient demographics
            keywords: Additional keywords
            
        Returns:
            Formatted query string
        """
        query_parts = []
        
        # Map conditions to clinical trial terminology
        condition_map = {
            'DIABETES': 'diabetes mellitus',
            'CARDIOVASCULAR': 'cardiovascular disease',
            'HYPERTENSION': 'hypertension',
            'RESPIRATORY': 'respiratory disease',
            'RENAL': 'kidney disease',
            'SEPSIS': 'sepsis',
            'NEUROLOGICAL': 'neurological disorder',
            'CANCER': 'cancer',
            'TRAUMA': 'trauma'
        }
        
        if conditions:
            mapped_conditions = [condition_map.get(c.upper(), c) for c in conditions]
            query_parts.extend(mapped_conditions)
        
        if demographics:
            age_range = demographics.get('age_range')
            if age_range:
                min_age, max_age = age_range
                if min_age >= 65:
                    query_parts.append('elderly')
                elif max_age <= 18:
                    query_parts.append('pediatric')
        
        if keywords:
            query_parts.extend(keywords)
        
        return ' AND '.join(query_parts) if query_parts else 'healthcare'
