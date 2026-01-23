"""
AI-BOM Generator — CycloneDX manifest for supply chain transparency

Generates a Software Bill of Materials (SBOM) for the AI system including:
- Model provenance
- Training data sources
- Library dependencies
"""
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelComponent:
    """AI model component."""
    name: str
    version: str
    publisher: str
    purl: str  # Package URL
    hashes: dict


@dataclass
class DatasetComponent:
    """Training dataset component."""
    name: str
    source: str
    size: int
    hash: str
    license: str


@dataclass
class AIBOM:
    """AI Bill of Materials (CycloneDX format)."""
    bom_format: str = "CycloneDX"
    spec_version: str = "1.5"
    version: int = 1
    serial_number: str = ""
    metadata: dict = None
    components: List[dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {
                "timestamp": datetime.now().isoformat(),
                "tools": [{"name": "securecli-aibom", "version": "1.0.0"}]
            }
        if self.components is None:
            self.components = []


class AIBOMGenerator:
    """Generate CycloneDX AI-BOM manifests."""
    
    def __init__(self):
        self.bom = AIBOM()
    
    def add_model(
        self,
        name: str,
        version: str,
        publisher: str = "Unknown",
        purl: str = "",
        sha256: str = ""
    ):
        """Add model component to BOM."""
        self.bom.components.append({
            "type": "machine-learning-model",
            "name": name,
            "version": version,
            "publisher": publisher,
            "purl": purl,
            "hashes": [{"alg": "SHA-256", "content": sha256}] if sha256 else []
        })
    
    def add_dataset(
        self,
        name: str,
        source: str,
        size: int = 0,
        sha256: str = "",
        license_: str = "Unknown"
    ):
        """Add training dataset to BOM."""
        self.bom.components.append({
            "type": "data",
            "name": name,
            "description": f"Training dataset from {source}",
            "properties": [
                {"name": "size", "value": str(size)},
                {"name": "license", "value": license_}
            ],
            "hashes": [{"alg": "SHA-256", "content": sha256}] if sha256 else []
        })
    
    def add_library(self, name: str, version: str, purl: str = ""):
        """Add library dependency to BOM."""
        self.bom.components.append({
            "type": "library",
            "name": name,
            "version": version,
            "purl": purl or f"pkg:pypi/{name}@{version}"
        })
    
    def generate(self, output_path: Optional[Path] = None) -> dict:
        """Generate CycloneDX BOM."""
        import uuid
        self.bom.serial_number = f"urn:uuid:{uuid.uuid4()}"
        
        bom_dict = {
            "bomFormat": self.bom.bom_format,
            "specVersion": self.bom.spec_version,
            "version": self.bom.version,
            "serialNumber": self.bom.serial_number,
            "metadata": self.bom.metadata,
            "components": self.bom.components
        }
        
        if output_path:
            with open(output_path, "w") as f:
                json.dump(bom_dict, f, indent=2)
        
        return bom_dict


def generate_securecli_bom():
    """Generate AI-BOM for SecureCLI-Tuner."""
    gen = AIBOMGenerator()
    
    # Add model
    gen.add_model(
        name="Qwen2.5-Coder-7B-Instruct",
        version="2.5",
        publisher="Alibaba",
        purl="pkg:huggingface/Qwen/Qwen2.5-Coder-7B-Instruct"
    )
    
    # Add datasets
    gen.add_dataset(
        name="prabhanshubhowal/natural_language_to_linux",
        source="HuggingFace",
        size=35000,
        license_="MIT"
    )
    gen.add_dataset(
        name="TLDR-pages",
        source="GitHub",
        license_="MIT"
    )
    
    # Add key libraries
    gen.add_library("transformers", "4.40.0")
    gen.add_library("peft", "0.10.0")
    gen.add_library("pysigma", "0.11.0")
    gen.add_library("mitreattack-python", "3.0.0")
    
    # Generate
    output = Path("compliance/cyclonedx/securecli-bom.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    gen.generate(output)
    
    print(f"Generated AI-BOM: {output}")
    return gen.bom


if __name__ == "__main__":
    generate_securecli_bom()
