"""
Dataset schemas for data pipeline validation.

Migrated from CLI-Tuner V1.
"""
from pydantic import BaseModel, Field


class BashCommandExample(BaseModel):
    """Schema for raw dataset examples (Bash)."""

    instruction: str = Field(min_length=3, max_length=500)
    input: str = Field(default="", max_length=200)
    output: str = Field(min_length=1, max_length=500)


class GitCommandExample(BaseModel):
    """Schema for Git command examples."""

    instruction: str = Field(min_length=3, max_length=500)
    input: str = Field(default="", max_length=200)
    output: str = Field(min_length=1, max_length=500)
    tool: str = Field(default="git")


class DockerCommandExample(BaseModel):
    """Schema for Docker command examples."""

    instruction: str = Field(min_length=3, max_length=500)
    input: str = Field(default="", max_length=200)
    output: str = Field(min_length=1, max_length=500)
    tool: str = Field(default="docker")
