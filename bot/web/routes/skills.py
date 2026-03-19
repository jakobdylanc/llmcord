"""Skills API endpoints for reading skill markdown files."""

import logging
import os
import re
from typing import Optional, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["skills"])

# Path to skills directory
SKILLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "llm", "tools", "skills")


class SkillParameter(BaseModel):
    """Parameter definition for a skill."""
    name: str
    type: str
    required: bool
    description: str


class SkillInfo(BaseModel):
    """Skill file info."""
    name: str
    filename: str
    description: Optional[str] = None
    parameters: list[SkillParameter] = []


class SkillDetail(BaseModel):
    """Full skill documentation."""
    name: str
    filename: str
    content: str
    word_count: int
    parameters: list[SkillParameter] = []


def _extract_skill_description(content: str) -> Optional[str]:
    """Extract description from skill markdown (first paragraph after title)."""
    if not content:
        return None
    
    # Skip YAML frontmatter if present
    lines = content.strip().split('\n')
    start_idx = 0
    
    # Check for YAML frontmatter (--- at start)
    if len(lines) > 0 and lines[0].strip() == '---':
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '---':
                start_idx = i + 1
                break
    
    in_description = False
    
    for line in lines[start_idx:]:
        line = line.strip()
        
        # Skip title lines
        if line.startswith('# '):
            continue
        
        # Look for description line (non-empty, not a code block or list)
        if line and not line.startswith('```') and not line.startswith('-') and not line.startswith('|'):
            # Return first meaningful line as description
            return line[:100] if len(line) > 100 else line
    
    return None


def _extract_parameters_from_signature(content: str) -> list[SkillParameter]:
    """Extract parameters from tool signature like `funcName(param1: type, param2: type = default)`"""
    parameters = []
    
    import re
    
    # First, extract content inside code blocks to avoid matching random text
    # Look for code blocks (```...```)
    code_blocks = re.findall(r'```[\w]*\n(.*?)```', content, re.DOTALL)
    
    # Also check for inline code `func()` patterns
    signature_content = '\n'.join(code_blocks)
    
    if not signature_content:
        return parameters
    
    # Match function signature: name(type1, type2) or name(type1, type2 = value)
    # The -> return_type is optional (some skills don't have it like google_tools)
    sig_pattern = re.search(r'(\w+)\s*\(([^)]+)\)(?:\s*->.*)?$', signature_content, re.MULTILINE)
    if not sig_pattern:
        return parameters
    
    params_str = sig_pattern.group(2)
    
    # Parse each parameter
    # Split by comma but handle default values with commas inside
    param_parts = []
    current = ""
    paren_depth = 0
    
    for char in params_str:
        if char == '(' or char == '[':
            paren_depth += 1
            current += char
        elif char == ')' or char == ']':
            paren_depth -= 1
            current += char
        elif char == ',' and paren_depth == 0:
            param_parts.append(current.strip())
            current = ""
        else:
            current += char
    
    if current.strip():
        param_parts.append(current.strip())
    
    for param in param_parts:
        param = param.strip()
        if not param:
            continue
        
        # Parse: name: type = default or name: type
        # Handle optional indicators like (optional) or [optional]
        optional_match = re.match(r'(\w+)\s*:\s*(\w+(?:\[\])?)\s*(?:=\s*[\'"]?([^,\'"]+)[\'"]?)?', param)
        
        if optional_match:
            name = optional_match.group(1)
            param_type = optional_match.group(2)
            default = optional_match.group(3)
            
            # Check if marked as optional
            is_optional = default is not None or '(optional)' in param.lower() or '[optional]' in param.lower()
            
            parameters.append(SkillParameter(
                name=name,
                type=param_type,
                required=not is_optional,
                description=f"Parameter{ ' (optional)' if is_optional else '' }"
            ))
    
    return parameters


def _extract_parameters(content: str) -> list[SkillParameter]:
    """Extract parameters from skill markdown."""
    parameters = []
    
    # Find the Parameters table in the markdown
    # Look for table with | Name | Type | Required | Description |
    param_section_match = re.search(
        r'##\s*Parameters\s*\n(\|.*\n)+',
        content,
        re.MULTILINE
    )
    
    if not param_section_match:
        return parameters
    
    # Parse the table
    lines = param_section_match.group(0).strip().split('\n')
    # Skip the header line (first |---...|) and process data rows
    in_data_rows = False
    for line in lines:
        if line.startswith('|') and '---' in line:
            in_data_rows = True
            continue
        if not in_data_rows or not line.startswith('|'):
            continue
        
        # Parse table row: | name | type | required | description | (4 columns)
        # OR: | Parameter | Type | Description | (3 columns - all required)
        parts = [p.strip() for p in line.split('|')[1:-1]]  # Skip first and last empty parts
        
        if len(parts) >= 4:
            # 4-column format: Name | Type | Required | Description
            name = parts[0]
            param_type = parts[1].lower()
            required = parts[2].lower() in ['yes', 'required', 'true', 'y']
            description = parts[3]
            
            parameters.append(SkillParameter(
                name=name,
                type=param_type,
                required=required,
                description=description
            ))
        elif len(parts) >= 3:
            # 3-column format: Parameter | Type | Description (assume all required)
            name = parts[0]
            param_type = parts[1].lower()
            description = parts[2]
            
            parameters.append(SkillParameter(
                name=name,
                type=param_type,
                required=True,
                description=description
            ))
    
    return parameters


@router.get("/skills", response_model=list[SkillInfo])
async def get_skills() -> list[SkillInfo]:
    """
    Get list of all available skills.
    
    Returns skill names, descriptions, and parameters.
    """
    skills = []
    
    if not os.path.exists(SKILLS_DIR):
        logger.warning(f"Skills directory not found: {SKILLS_DIR}")
        return []
    
    for filename in os.listdir(SKILLS_DIR):
        if filename.endswith('.md'):
            filepath = os.path.join(SKILLS_DIR, filename)
            
            # Read first part of file for description
            description = None
            parameters = []
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read(2000)  # Read more to get parameters too
                    description = _extract_skill_description(content)
                    # Try table first, then fall back to signature
                    parameters = _extract_parameters(content)
                    if not parameters:
                        parameters = _extract_parameters_from_signature(content)
            except Exception as e:
                logger.warning(f"Error reading skill {filename}: {e}")
            
            # Name from filename (without .md extension)
            name = filename[:-3]
            
            skills.append(SkillInfo(
                name=name,
                filename=filename,
                description=description,
                parameters=parameters,
            ))
    
    # Sort by name
    skills.sort(key=lambda s: s.name)
    
    return skills


@router.get("/skills/{name}", response_model=SkillDetail)
async def get_skill(name: str) -> SkillDetail:
    """
    Get full skill documentation by name.
    
    - **name**: Skill name (without .md extension)
    """
    filename = f"{name}.md"
    filepath = os.path.join(SKILLS_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract parameters from full content (table first, then signature fallback)
        parameters = _extract_parameters(content)
        if not parameters:
            parameters = _extract_parameters_from_signature(content)
        
        return SkillDetail(
            name=name,
            filename=filename,
            content=content,
            word_count=len(content.split()),
            parameters=parameters,
        )
    except Exception as e:
        logger.error(f"Error reading skill {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading skill: {str(e)}")