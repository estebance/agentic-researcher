from pydantic import BaseModel, Field


class ClassificationModel(BaseModel):
    category: str = Field(
        description="Category the text belongs to",
        enum=["positive", "negative", "neutral"]
    )