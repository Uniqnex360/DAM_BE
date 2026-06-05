from pydantic import BaseModel, ConfigDict, BeforeValidator
from uuid import UUID
from typing import Optional, List, Annotated
from datetime import datetime
from decimal import Decimal

def to_float(v):
    if v is None: return 0.0
    if isinstance(v, (Decimal, int, float)): return float(v)
    return v

def to_int(v):
    if v is None: return 0
    if isinstance(v, (Decimal, float, int)): return int(float(v))
    return v

SafeFloat = Annotated[float, BeforeValidator(to_float)]
SafeInt = Annotated[int, BeforeValidator(to_int)]

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    destinations: Optional[str] = None

class ProjectResponse(BaseModel):
    id: UUID 
    name: str
    client: Optional[str] = None
    use_case: Optional[str] = None
    operation_mode: Optional[str] = None
    source_status: Optional[str] = None
    created_at: Optional[datetime] = None
    import_file_name: Optional[str] = None
    source_processing_status: Optional[str] = None
    
    completeness_score: Optional[SafeFloat] = 0.0
    data_quality_score: Optional[SafeFloat] = 0.0
    
    algorithm_used: Optional[str] = None 
    product_count: SafeInt = 0
    aggregated_count: SafeInt = 0
    aggregation_type: Optional[str] = None
    processing_status: str = 'pending'
    cleaned_count: SafeInt = 0
    failed_count: SafeInt = 0
    enrichment_pending_count: SafeInt = 0
    pending_count: SafeInt = 0

    model_config = ConfigDict(from_attributes=True)

class ProjectListResponse(BaseModel):
    total: int
    page: int
    limit: int
    projects: List[ProjectResponse]