from typing import Dict, Any, Optional
from pydantic import BaseModel


class FormFieldOption(BaseModel):
    """A single option for select/checkbox fields."""
    label: str
    value: str
    example: Optional[Dict[str, Any]] = None


class FormFieldOptionsRequest(BaseModel):
    """Request body for getting field options."""
    field_type: str
    field_config: Dict[str, Any] = {}


class FormValidationRequest(BaseModel):
    """Request body for form validation."""
    form_schema: Dict[str, Any]
    form_data: Dict[str, Any]
