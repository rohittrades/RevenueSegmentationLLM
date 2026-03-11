from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
# Pydantic Model

class LocationScope(str, Enum):
    INTERNATIONAL = "International"
    DOMESTIC = "Domestic"
    BOTH = "Both"

class VerticalStatus(str, Enum):
    JUST_LAUNCHED = "Just Launched"
    EMERGING = "Emerging"
    ESTABLISHED = "Established"

class VerticalDetail(BaseModel):

    # business_segment and products_or_services
    business_segment: str = Field(
        ..., 
        description="The business vertical of the company or business segment or a superficial category grouping similar products or services under same umbrella."
    )
    segment_summary: str = Field(
        ...,
        description=(
            "Objective 25-30 word description of core operations and technical USP. "
            "Strictly avoid 'leadership' claims and marketing adjectives. "
            "Example: 'Manufactures synthetic organic chemicals and reagents for pharmaceutical labs, "
            "utilizing high-pressure hydrogenation and cryogenic reaction capabilities.'"
        )
    )
    products_or_services: List[str] = Field(
        ..., 
        description="List of specific products or service lines under this business vertical."
    )

    # Location information
    operational_locations: List[str] = Field(
        ..., 
        description=(
            "List of cities, states, or countries where physical assets (plants/offices) are located. "
            "If the information is available, list them in order of production capacity or importance."
        )
    )
    has_domestic_presence: bool = Field(
        ..., 
        description="True if the business vertical serves to Indias domestic market."
    )
    domestic_reach_states: List[str] = Field(
        default_factory=list,
        description=(
            "List of domestic states or regions served. If data allows, order by market share or "
            "revenue contribution. Use ['Pan-India'] if specific regions aren't disclosed."
        )
    )
    has_international_presence: bool = Field(
        ..., 
        description="True if the business vertical exports to or operates in foreign markets."
    )
    international_markets: List[str] = Field(
        default_factory=list,
        description=(
            "List of countries or global regions served. If the information is available, "
            "list in order of strategic importance or export volume."
        )
    )

    # Commission year
    commission_year: Optional[int] = Field(
        None, 
        ge=1800, 
        le=2026, 
        description="The year the first major facility for this vertical opened. Use null if unknown.")
    
    status: VerticalStatus = Field(..., description="Maturity based on years active and market presence.")

    # CapEx Information
    capex_status: bool = Field(
        ..., 
        description="True if the vertical has recently announced or is currently undergoing capital expenditure; False otherwise."
    )

    capex_value_inr_crores: Optional[float] = Field(
        None,
        description="Total investment amount in INR Crores. Provide the number only. Leave null if not disclosed."
    )

    capex_details: Optional[str] = Field(
        None,
        description=(
            "A concise summary including: 1) Purpose (e.g., expansion, R&D), 2) Location, "
            "3) Capacity added (with units), and 4) Target product/service. "
            "Leave null if capex_status is False."
        )
    )

    # Revenue Information
    is_segment_revenue_disclosed: bool = Field(
        ...,
        description="True if the revenue value was specifically reported for this segment; False if you are seeing only group-level data."
    )

    segment_revenue: Optional[float] = Field(
        None, 
        description=(
            "Revenue for the current business segment in INR Crores." 
            "Leave null if not disclosed or unclear "
        )
    )
    
    previous_year_revenue: Optional[float] = Field(
        None, 
        description=(
            "Previous year revenue for the current business segment in INR Crores." 
            "Leave null if not disclosed or unclear "
        )
    )

    # Top Clients Information
    top_clients: List[str] = Field(
        default_factory=list,
        description=(
            "List of key customers or client names. If specific names are not disclosed, "
            "list the types of clients (e.g., 'Major PSU Banks', 'Leading Automobile OEMs'). "
            "List in order of importance/revenue contribution if known."
        )
    )

    client_concentration: Optional[str] = Field(
        None,
        description=(
            "Briefly describe the dependency on top clients. "
            "Example: 'Top 5 clients contribute 60% of segment revenue' or 'Highly diversified client base'."
        )
    )

class CompanyAnalysis(BaseModel):
    company_name: str = Field(..., description="Full legal name of the company being analyzed.")
    verticals: List[VerticalDetail] = Field(..., description="A list of consolidated business segments.")