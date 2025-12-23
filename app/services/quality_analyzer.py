import base64
import numpy as np
import cv2
from app.services.image_processor import ImageProcessor
from app.schemas.analysis import AnalysisResult, Suggestions, Issue, Compliance, ComplianceCheck


def analyze_image_quality(request_data) -> AnalysisResult:

    try:
        if "," in request_data.imageBase64:
            encoded_data = request_data.imageBase64.split(',')[1]
        else:
            encoded_data = request_data.imageBase64
        file_bytes = base64.b64decode(encoded_data)
    except:
        raise ValueError("Invalid Base64 string")

    processor = ImageProcessor(file_bytes)
    conf = processor.analyze()

    score = 100
    issues = []

    bg_is_dirty = conf["bg_clean"] > 0.6
    if bg_is_dirty:
        score -= 20
        issues.append(Issue(type="Background", severity="medium",
                      description="Cluttered background detected", suggestedAction="Remove Background"))

    has_shadows = conf["shadow"] > 0.5
    if has_shadows:
        score -= 15
        issues.append(Issue(type="Lighting", severity="medium",
                      description="Harsh shadows detected", suggestedAction="Shadow Correction"))

    is_small = request_data.width < 1000 or request_data.height < 1000
    if is_small:
        score -= 15
        issues.append(Issue(type="Resolution", severity="high",
                      description="Image resolution is low", suggestedAction="Upscale"))

    amazon_violations = []
    if bg_is_dirty:
        amazon_violations.append("Background must be pure white")
    if is_small:
        amazon_violations.append("1000px minimum required")

    return AnalysisResult(
        qualityScore=max(0, score),
        productCategory="Product",
        backgroundAnalysis={"type": "Complex" if bg_is_dirty else "Clean"},
        suggestions=Suggestions(
            backgroundRemoval=bg_is_dirty,
            upscaling=is_small,
            cropping=conf["crop"] > 0.5,
            enhancement=has_shadows,
            compression=request_data.fileSize > 2 * 1024 * 1024
        ),
        issues=issues,
        compliance=Compliance(
            amazon=ComplianceCheck(isCompliant=len(
                amazon_violations) == 0, violations=amazon_violations),
            shopify=ComplianceCheck(isCompliant=True, violations=[])
        )
    )
