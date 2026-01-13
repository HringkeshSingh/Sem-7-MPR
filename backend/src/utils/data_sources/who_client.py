"""
WHO (World Health Organization) data client.

Retrieves health data, guidelines, and statistics from WHO APIs and databases.
"""

import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .base_client import BaseDataSourceClient, RetrievedDocument

logger = logging.getLogger(__name__)


class WHOClient(BaseDataSourceClient):
    """
    Client for WHO Global Health Observatory (GHO) API.
    
    Retrieves:
    - Health indicators
    - Disease statistics
    - Global health guidelines
    """
    
    GHO_API_URL = "https://ghoapi.azureedge.net/api"
    
    # Map conditions to WHO indicator codes
    INDICATOR_MAP = {
        'DIABETES': ['NCD_GLUC_05', 'NCD_GLUC_04'],
        'CARDIOVASCULAR': ['NCD_HYP_TREATMENT', 'NCDMORT3070'],
        'HYPERTENSION': ['NCD_HYP_PREVALENCE_A', 'BP_04'],
        'RESPIRATORY': ['AIR_1', 'TOBACCO_0000000192'],
        'CANCER': ['NCDMORT3070CANCER', 'CERVICAL_CANCER'],
        'MORTALITY': ['LIFE_0000000029', 'LIFE_0000000030']
    }
    
    def __init__(self, rate_limit: float = 1.0):
        """Initialize the WHO client."""
        super().__init__(source_name="who", rate_limit=rate_limit)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HealthcareDataGenerator/1.0'
        })
    
    def _check_availability(self) -> bool:
        """Check if WHO API is accessible."""
        try:
            response = self.session.get(
                f"{self.GHO_API_URL}/Indicator",
                params={"$top": 1},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"WHO API not available: {e}")
            return False
    
    def search(self, query: str, max_results: int = 20) -> List[RetrievedDocument]:
        """
        Search WHO data for health information.
        
        Args:
            query: Search query (conditions, health topics)
            max_results: Maximum number of results
            
        Returns:
            List of RetrievedDocument objects
        """
        self._rate_limit_wait()
        documents = []
        
        # Search indicators
        try:
            indicator_docs = self._search_indicators(query, max_results // 2)
            documents.extend(indicator_docs)
        except Exception as e:
            logger.warning(f"Error searching WHO indicators: {e}")
        
        # Get condition-specific data
        try:
            condition_docs = self._get_condition_data(query, max_results // 2)
            documents.extend(condition_docs)
        except Exception as e:
            logger.warning(f"Error getting WHO condition data: {e}")
        
        logger.info(f"Retrieved {len(documents)} WHO documents for query: {query}")
        return documents[:max_results]
    
    def _search_indicators(self, query: str, max_results: int) -> List[RetrievedDocument]:
        """Search WHO indicators matching the query."""
        try:
            # Search for indicators
            response = self.session.get(
                f"{self.GHO_API_URL}/Indicator",
                params={
                    "$filter": f"contains(IndicatorName,'{query}')",
                    "$top": max_results
                },
                timeout=30
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            indicators = data.get("value", [])
            
            documents = []
            for i, indicator in enumerate(indicators):
                doc = self._parse_indicator(indicator, i, len(indicators))
                if doc:
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching WHO indicators: {e}")
            return []
    
    def _parse_indicator(self, indicator: Dict[str, Any], index: int, total: int) -> Optional[RetrievedDocument]:
        """Parse a WHO indicator into RetrievedDocument format."""
        try:
            indicator_code = indicator.get("IndicatorCode", "")
            indicator_name = indicator.get("IndicatorName", "Unknown Indicator")
            
            content = f"WHO Health Indicator: {indicator_name}\n\n"
            
            # Try to get indicator data
            try:
                data_response = self.session.get(
                    f"{self.GHO_API_URL}/{indicator_code}",
                    params={"$top": 10, "$orderby": "TimeDim desc"},
                    timeout=15
                )
                
                if data_response.status_code == 200:
                    data = data_response.json()
                    values = data.get("value", [])[:5]
                    
                    if values:
                        content += "Recent Data:\n"
                        for v in values:
                            country = v.get("SpatialDim", "Unknown")
                            year = v.get("TimeDim", "Unknown")
                            value = v.get("NumericValue", "N/A")
                            content += f"- {country} ({year}): {value}\n"
            except Exception:
                pass
            
            metadata = {
                "indicator_code": indicator_code,
                "indicator_name": indicator_name,
                "category": "health_indicator"
            }
            
            return RetrievedDocument(
                source="who",
                source_id=indicator_code,
                title=indicator_name,
                content=content,
                url=f"https://www.who.int/data/gho/data/indicators/indicator-details/GHO/{indicator_code}",
                metadata=metadata,
                relevance_score=1.0 - (index / max(total, 1))
            )
            
        except Exception as e:
            logger.warning(f"Error parsing WHO indicator: {e}")
            return None
    
    def _get_condition_data(self, query: str, max_results: int) -> List[RetrievedDocument]:
        """Get WHO data for specific health conditions."""
        documents = []
        
        # Check if query matches any known conditions
        query_upper = query.upper()
        matching_conditions = []
        
        for condition, indicators in self.INDICATOR_MAP.items():
            if condition in query_upper or any(word in query_upper for word in condition.split('_')):
                matching_conditions.append((condition, indicators))
        
        for condition, indicator_codes in matching_conditions[:2]:  # Limit to 2 conditions
            for code in indicator_codes[:1]:  # Limit to 1 indicator per condition
                try:
                    response = self.session.get(
                        f"{self.GHO_API_URL}/{code}",
                        params={"$top": 20, "$orderby": "TimeDim desc"},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        values = data.get("value", [])
                        
                        if values:
                            doc = self._create_condition_document(condition, code, values)
                            if doc:
                                documents.append(doc)
                                
                except Exception as e:
                    logger.warning(f"Error fetching WHO data for {code}: {e}")
        
        return documents[:max_results]
    
    def _create_condition_document(
        self, 
        condition: str, 
        indicator_code: str, 
        values: List[Dict]
    ) -> Optional[RetrievedDocument]:
        """Create a document from condition-specific WHO data."""
        try:
            title = f"WHO Global Data: {condition.replace('_', ' ').title()}"
            
            content_parts = [f"Global Health Statistics for {condition.replace('_', ' ').title()}\n"]
            
            # Group by country
            by_country = {}
            for v in values[:50]:
                country = v.get("SpatialDim", "Unknown")
                year = v.get("TimeDim", "Unknown")
                value = v.get("NumericValue")
                
                if value is not None:
                    if country not in by_country:
                        by_country[country] = []
                    by_country[country].append((year, value))
            
            # Create summary
            for country, data in list(by_country.items())[:10]:
                latest = data[0]
                content_parts.append(f"- {country}: {latest[1]} ({latest[0]})")
            
            content = "\n".join(content_parts)
            
            return RetrievedDocument(
                source="who",
                source_id=f"{condition}_{indicator_code}",
                title=title,
                content=content,
                url="https://www.who.int/data/gho",
                metadata={
                    "condition": condition,
                    "indicator_code": indicator_code,
                    "data_type": "statistics",
                    "countries_covered": list(by_country.keys())[:10]
                },
                relevance_score=0.8
            )
            
        except Exception as e:
            logger.warning(f"Error creating condition document: {e}")
            return None
    
    def fetch_by_id(self, indicator_code: str) -> Optional[RetrievedDocument]:
        """Fetch a specific WHO indicator by code."""
        self._rate_limit_wait()
        
        try:
            # Get indicator info
            response = self.session.get(
                f"{self.GHO_API_URL}/Indicator",
                params={"$filter": f"IndicatorCode eq '{indicator_code}'"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                indicators = data.get("value", [])
                if indicators:
                    return self._parse_indicator(indicators[0], 0, 1)
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching WHO indicator {indicator_code}: {e}")
            return None
