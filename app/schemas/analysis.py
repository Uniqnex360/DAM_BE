from pydantic import BaseModel
from typing import List, Optional, Dict

class Issue(BaseModel):
    type: str
    severity: str  # 'high', 'medium', 'low'
    description: str
    suggestedAction: str
class ComplianceCheck(BaseModel):
    isCompliant: bool
    violations: List[str]


class ComplianceDetails(BaseModel):
    isCompliant: bool
    violations: List[str]

class Compliance(BaseModel):
    amazon: ComplianceDetails
    shopify: ComplianceDetails

class Suggestions(BaseModel):
    backgroundRemoval: bool
    upscaling: bool
    cropping: bool
    enhancement: bool
    compression: bool
    
class AnalysisResult(BaseModel):
    qualityScore: int
    productCategory: str
    backgroundAnalysis: Dict[str, str]
    suggestions: Suggestions
    issues: List[Issue]
    compliance: Compliance
    
class BackgroundAnalysis(BaseModel):
    type: str

class ImageAnalysis(BaseModel):
    qualityScore: int
    productCategory: str
    backgroundAnalysis: BackgroundAnalysis
    suggestions: Suggestions
    issues: List[Issue]
    compliance: Compliance

# The Request Body from Frontend
class AnalyzeRequest(BaseModel):
    imageBase64: str
    fileName: str
    fileSize: int
    width: int
    height: int

class AnalyzeResponse(BaseModel):
    analysis: ImageAnalysis