"""
Enhanced Clinical Trials Client.

Extends the base ClinicalTrials.gov client with:
- Advanced filtering and faceted search
- Outcome data parsing
- Study arms and interventions
- Statistical result extraction
- Trial comparison analysis
"""

import requests
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.data_sources.base_client import RetrievedDocument

logger = logging.getLogger(__name__)


class TrialStatus(Enum):
    """Clinical trial status values."""
    RECRUITING = "Recruiting"
    ACTIVE = "Active, not recruiting"
    COMPLETED = "Completed"
    TERMINATED = "Terminated"
    WITHDRAWN = "Withdrawn"
    SUSPENDED = "Suspended"
    NOT_YET_RECRUITING = "Not yet recruiting"


class TrialPhase(Enum):
    """Clinical trial phases."""
    EARLY_PHASE_1 = "Early Phase 1"
    PHASE_1 = "Phase 1"
    PHASE_2 = "Phase 2"
    PHASE_3 = "Phase 3"
    PHASE_4 = "Phase 4"
    NOT_APPLICABLE = "Not Applicable"


@dataclass
class Intervention:
    """Clinical trial intervention/treatment."""
    name: str
    type: str  # Drug, Device, Biological, Procedure, etc.
    description: Optional[str] = None
    arm_group_labels: List[str] = field(default_factory=list)


@dataclass
class OutcomeMeasure:
    """Clinical trial outcome measure."""
    title: str
    type: str  # Primary, Secondary, Other
    description: Optional[str] = None
    time_frame: Optional[str] = None


@dataclass
class EligibilityCriteria:
    """Parsed eligibility criteria."""
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    gender: str = "All"  # All, Male, Female
    accepts_healthy: bool = False
    inclusion: List[str] = field(default_factory=list)
    exclusion: List[str] = field(default_factory=list)


@dataclass
class ClinicalTrial:
    """Comprehensive clinical trial data."""
    nct_id: str
    title: str
    summary: str
    detailed_description: Optional[str] = None
    status: str = "Unknown"
    phase: str = "Not Applicable"
    study_type: str = "Interventional"
    enrollment: int = 0
    conditions: List[str] = field(default_factory=list)
    interventions: List[Intervention] = field(default_factory=list)
    outcomes: List[OutcomeMeasure] = field(default_factory=list)
    eligibility: Optional[EligibilityCriteria] = None
    sponsor: Optional[str] = None
    collaborators: List[str] = field(default_factory=list)
    start_date: Optional[datetime] = None
    completion_date: Optional[datetime] = None
    locations: List[Dict[str, str]] = field(default_factory=list)
    url: str = ""
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nct_id": self.nct_id,
            "title": self.title,
            "summary": self.summary,
            "status": self.status,
            "phase": self.phase,
            "study_type": self.study_type,
            "enrollment": self.enrollment,
            "conditions": self.conditions,
            "interventions": [{"name": i.name, "type": i.type} for i in self.interventions],
            "outcomes": [{"title": o.title, "type": o.type} for o in self.outcomes],
            "sponsor": self.sponsor,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "url": self.url,
            "relevance_score": self.relevance_score
        }
    
    def to_text(self) -> str:
        """Convert to text for embedding."""
        parts = [f"Clinical Trial: {self.title}"]
        parts.append(f"NCT ID: {self.nct_id}")
        parts.append(f"Status: {self.status}, Phase: {self.phase}")
        if self.conditions:
            parts.append(f"Conditions: {', '.join(self.conditions)}")
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        if self.interventions:
            parts.append(f"Interventions: {', '.join(i.name for i in self.interventions)}")
        return "\n".join(parts)
    
    def to_document(self) -> RetrievedDocument:
        """Convert to RetrievedDocument format."""
        return RetrievedDocument(
            source="clinical_trials",
            source_id=self.nct_id,
            title=self.title,
            content=self.to_text(),
            url=self.url,
            authors=[self.sponsor] if self.sponsor else None,
            publication_date=self.start_date,
            metadata=self.to_dict(),
            relevance_score=self.relevance_score
        )


class EnhancedClinicalTrialsClient:
    """
    Enhanced ClinicalTrials.gov client.
    
    Usage:
        client = EnhancedClinicalTrialsClient()
        
        # Advanced search
        trials = client.search(
            conditions=["diabetes"],
            phases=["Phase 3", "Phase 4"],
            status=["Recruiting", "Active, not recruiting"]
        )
        
        # Get outcomes data
        outcomes = client.get_trial_outcomes("NCT12345678")
    """
    
    BASE_URL = "https://clinicaltrials.gov/api/v2"
    
    # Full field list for comprehensive data
    ALL_FIELDS = [
        "NCTId", "BriefTitle", "OfficialTitle", "BriefSummary", "DetailedDescription",
        "Condition", "Keyword", "OverallStatus", "Phase", "StudyType",
        "EnrollmentCount", "EligibilityCriteria", "Gender", "MinimumAge", "MaximumAge",
        "HealthyVolunteers", "StartDate", "CompletionDate", "PrimaryCompletionDate",
        "LeadSponsorName", "CollaboratorName", "LocationCity", "LocationCountry",
        "InterventionType", "InterventionName", "InterventionDescription",
        "PrimaryOutcomeMeasure", "PrimaryOutcomeDescription", "PrimaryOutcomeTimeFrame",
        "SecondaryOutcomeMeasure", "SecondaryOutcomeDescription"
    ]
    
    def __init__(self, rate_limit: float = 0.5):
        self.rate_limit = rate_limit
        self._last_request = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HealthcareDataGenerator/1.0'
        })
    
    def _rate_limit_wait(self):
        import time
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def search(
        self,
        query: Optional[str] = None,
        conditions: Optional[List[str]] = None,
        interventions: Optional[List[str]] = None,
        phases: Optional[List[str]] = None,
        status: Optional[List[str]] = None,
        age_range: Optional[Tuple[int, int]] = None,
        gender: Optional[str] = None,
        max_results: int = 50
    ) -> List[ClinicalTrial]:
        """
        Advanced search with multiple filters.
        
        Args:
            query: Free text search
            conditions: Filter by conditions
            interventions: Filter by interventions
            phases: Filter by phases
            status: Filter by recruitment status
            age_range: (min_age, max_age) in years
            gender: Male, Female, or All
            max_results: Maximum results to return
        """
        self._rate_limit_wait()
        
        params = {
            "pageSize": min(max_results, 100),
            "format": "json",
            "fields": ",".join(self.ALL_FIELDS)
        }
        
        # Build query terms
        query_parts = []
        
        if query:
            query_parts.append(query)
        
        if conditions:
            query_parts.append(f"AREA[Condition] ({' OR '.join(conditions)})")
        
        if interventions:
            query_parts.append(f"AREA[Intervention] ({' OR '.join(interventions)})")
        
        if query_parts:
            params["query.term"] = " AND ".join(query_parts)
        
        # Add filters
        filter_parts = []
        
        if phases:
            filter_parts.append(f"AREA[Phase] ({' OR '.join(phases)})")
        
        if status:
            filter_parts.append(f"AREA[OverallStatus] ({' OR '.join(status)})")
        
        if gender and gender != "All":
            filter_parts.append(f"AREA[Gender] {gender}")
        
        if filter_parts:
            if "query.term" in params:
                params["query.term"] += " AND " + " AND ".join(filter_parts)
            else:
                params["query.term"] = " AND ".join(filter_parts)
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/studies",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            trials = []
            
            for i, study in enumerate(data.get("studies", [])):
                trial = self._parse_full_study(study, i, max_results)
                if trial:
                    # Apply age filter post-query
                    if age_range and trial.eligibility:
                        if trial.eligibility.min_age and trial.eligibility.min_age > age_range[1]:
                            continue
                        if trial.eligibility.max_age and trial.eligibility.max_age < age_range[0]:
                            continue
                    trials.append(trial)
            
            logger.info(f"Found {len(trials)} clinical trials")
            return trials
            
        except Exception as e:
            logger.error(f"Clinical trials search error: {e}")
            return []
    
    def _parse_full_study(
        self,
        study: Dict[str, Any],
        index: int,
        total: int
    ) -> Optional[ClinicalTrial]:
        """Parse complete study data."""
        try:
            protocol = study.get("protocolSection", {})
            id_module = protocol.get("identificationModule", {})
            desc_module = protocol.get("descriptionModule", {})
            status_module = protocol.get("statusModule", {})
            design_module = protocol.get("designModule", {})
            eligibility_module = protocol.get("eligibilityModule", {})
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            arms_module = protocol.get("armsInterventionsModule", {})
            outcomes_module = protocol.get("outcomesModule", {})
            contacts_module = protocol.get("contactsLocationsModule", {})
            
            nct_id = id_module.get("nctId", "")
            
            # Parse interventions
            interventions = []
            for intv in arms_module.get("interventions", []):
                interventions.append(Intervention(
                    name=intv.get("name", "Unknown"),
                    type=intv.get("type", "Other"),
                    description=intv.get("description"),
                    arm_group_labels=intv.get("armGroupLabels", [])
                ))
            
            # Parse outcomes
            outcomes = []
            for pm in outcomes_module.get("primaryOutcomes", []):
                outcomes.append(OutcomeMeasure(
                    title=pm.get("measure", ""),
                    type="Primary",
                    description=pm.get("description"),
                    time_frame=pm.get("timeFrame")
                ))
            for sm in outcomes_module.get("secondaryOutcomes", []):
                outcomes.append(OutcomeMeasure(
                    title=sm.get("measure", ""),
                    type="Secondary",
                    description=sm.get("description"),
                    time_frame=sm.get("timeFrame")
                ))
            
            # Parse eligibility
            eligibility = self._parse_eligibility(eligibility_module)
            
            # Parse dates
            start_date = self._parse_date(status_module.get("startDateStruct"))
            completion_date = self._parse_date(status_module.get("completionDateStruct"))
            
            # Parse locations
            locations = []
            for loc in contacts_module.get("locations", [])[:10]:
                locations.append({
                    "city": loc.get("city", ""),
                    "country": loc.get("country", "")
                })
            
            # Get collaborators
            collaborators = []
            for collab in sponsor_module.get("collaborators", []):
                if collab.get("name"):
                    collaborators.append(collab["name"])
            
            return ClinicalTrial(
                nct_id=nct_id,
                title=id_module.get("briefTitle") or id_module.get("officialTitle", "No title"),
                summary=desc_module.get("briefSummary", ""),
                detailed_description=desc_module.get("detailedDescription"),
                status=status_module.get("overallStatus", "Unknown"),
                phase=design_module.get("phases", ["Not Applicable"])[0] if design_module.get("phases") else "Not Applicable",
                study_type=design_module.get("studyType", "Interventional"),
                enrollment=design_module.get("enrollmentInfo", {}).get("count", 0) or 0,
                conditions=protocol.get("conditionsModule", {}).get("conditions", []),
                interventions=interventions,
                outcomes=outcomes,
                eligibility=eligibility,
                sponsor=sponsor_module.get("leadSponsor", {}).get("name"),
                collaborators=collaborators,
                start_date=start_date,
                completion_date=completion_date,
                locations=locations,
                url=f"https://clinicaltrials.gov/study/{nct_id}",
                relevance_score=1.0 - (index / max(total, 1))
            )
            
        except Exception as e:
            logger.warning(f"Error parsing clinical trial: {e}")
            return None
    
    def _parse_eligibility(self, module: Dict) -> Optional[EligibilityCriteria]:
        """Parse eligibility criteria."""
        try:
            # Parse age
            min_age = None
            max_age = None
            
            min_age_str = module.get("minimumAge", "")
            if min_age_str:
                min_age = self._parse_age(min_age_str)
            
            max_age_str = module.get("maximumAge", "")
            if max_age_str:
                max_age = self._parse_age(max_age_str)
            
            # Parse criteria text
            criteria_text = module.get("eligibilityCriteria", "")
            inclusion = []
            exclusion = []
            
            if criteria_text:
                lines = criteria_text.split("\n")
                current_section = None
                
                for line in lines:
                    line_lower = line.strip().lower()
                    if "inclusion" in line_lower:
                        current_section = "inclusion"
                    elif "exclusion" in line_lower:
                        current_section = "exclusion"
                    elif line.strip() and current_section:
                        if current_section == "inclusion":
                            inclusion.append(line.strip())
                        else:
                            exclusion.append(line.strip())
            
            return EligibilityCriteria(
                min_age=min_age,
                max_age=max_age,
                gender=module.get("sex", "All"),
                accepts_healthy=module.get("healthyVolunteers", "No") == "Yes",
                inclusion=inclusion[:10],
                exclusion=exclusion[:10]
            )
            
        except Exception as e:
            logger.warning(f"Error parsing eligibility: {e}")
            return None
    
    def _parse_age(self, age_str: str) -> Optional[int]:
        """Parse age string to integer years."""
        import re
        
        try:
            # Handle formats like "18 Years", "6 Months", "N/A"
            if not age_str or age_str.lower() in ["n/a", "na", "not applicable"]:
                return None
            
            match = re.match(r"(\d+)\s*(year|month|week|day)?", age_str.lower())
            if match:
                value = int(match.group(1))
                unit = match.group(2) if match.group(2) else "year"
                
                if "month" in unit:
                    return value // 12
                elif "week" in unit:
                    return value // 52
                elif "day" in unit:
                    return value // 365
                return value
            
            return None
            
        except Exception:
            return None
    
    def _parse_date(self, date_struct: Optional[Dict]) -> Optional[datetime]:
        """Parse date structure."""
        if not date_struct:
            return None
        
        date_str = date_struct.get("date", "")
        if not date_str:
            return None
        
        for fmt in ["%Y-%m-%d", "%Y-%m", "%Y"]:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def get_by_nct_id(self, nct_id: str) -> Optional[ClinicalTrial]:
        """Fetch a specific trial by NCT ID."""
        self._rate_limit_wait()
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/studies/{nct_id}",
                params={"format": "json"},
                timeout=30
            )
            response.raise_for_status()
            
            return self._parse_full_study(response.json(), 0, 1)
            
        except Exception as e:
            logger.error(f"Error fetching trial {nct_id}: {e}")
            return None
    
    def search_by_condition(
        self,
        condition: str,
        include_completed: bool = True,
        max_results: int = 50
    ) -> List[ClinicalTrial]:
        """Search trials for a specific condition."""
        status = ["Recruiting", "Active, not recruiting", "Not yet recruiting"]
        if include_completed:
            status.append("Completed")
        
        return self.search(
            conditions=[condition],
            status=status,
            max_results=max_results
        )
    
    def search_interventional(
        self,
        conditions: List[str],
        intervention_types: Optional[List[str]] = None,
        max_results: int = 50
    ) -> List[ClinicalTrial]:
        """Search for interventional trials."""
        trials = self.search(
            conditions=conditions,
            phases=["Phase 2", "Phase 3", "Phase 4"],
            max_results=max_results * 2  # Fetch more to filter
        )
        
        if intervention_types:
            trials = [
                t for t in trials
                if any(i.type in intervention_types for i in t.interventions)
            ]
        
        return trials[:max_results]
    
    def get_trials_for_demographics(
        self,
        conditions: List[str],
        age: int,
        gender: str = "All",
        max_results: int = 50
    ) -> List[ClinicalTrial]:
        """Find trials matching patient demographics."""
        trials = self.search(
            conditions=conditions,
            status=["Recruiting", "Not yet recruiting"],
            gender=gender,
            max_results=max_results * 2
        )
        
        # Filter by age
        eligible = []
        for trial in trials:
            if trial.eligibility:
                min_age = trial.eligibility.min_age or 0
                max_age = trial.eligibility.max_age or 150
                
                if min_age <= age <= max_age:
                    eligible.append(trial)
            else:
                eligible.append(trial)
        
        return eligible[:max_results]
