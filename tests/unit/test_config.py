import pytest
from pydantic import ValidationError
from config import validate_parametrization_file

def test_valid_parameters():
    valid_json = {
        "provider": "example_provider",
        "llm_model_id": "example_model_id",
        "kdb_id": "L1VS0KQQTR",
        "kdb_number_of_results": 3,
        "kdb_region": "us-east-1",
        "kdb_retriever_params": {
            "kdb_id": "L1VS0KQQTR",
            "kdb_max_number_of_results": 3,
            "kdb_region": "us-east-1",
        },
        "web_retriever": {
            "activated": False,
            "urls": ["http://example.com"],
            "is_advanced_search": True,
            "max_number_of_resources": 5
        },
        "checkpointer": {
            "endpoint": "127.0.0.1",
            "port": 6379,
            "db_number": 0
        }
    }

    parametrization = validate_parametrization_file(valid_json)
    assert parametrization.provider == "example_provider"
    assert parametrization.llm_model_id == "example_model_id"
    assert parametrization.web_retriever.urls == ["http://example.com"]
    assert parametrization.web_retriever.is_advanced_search is True

def test_invalid_parameters():
    invalid_json = {
        "provider": "example_provider",
        "llm_model_id": "example_model_id",
        "kdb_retriever_params": {
            "kdb_id": "L1VS0KQQTR",
            "kdb_max_number_of_results": 3,
            "kdb_region": "us-east-1",
        },
        "web_retriever": {
            "activated": False,
            "urls": "http://example.com",
            "is_advanced_search": True,
            "max_number_of_resources": 5
        },
        "checkpointer": {
            "endpoint": "127.0.0.1",
            "port": 6379,
            "db_number": 0
        }
    }

    with pytest.raises(ValidationError):
        validate_parametrization_file(invalid_json)

def test_missing_parameters():
    missing_json = {
        "provider": "example_provider",
        "llm_model_id": "example_model_id",
        "kdb_retriever_params": {
            "kdb_id": "L1VS0KQQTR",
            "kdb_max_number_of_results": 3,
            "kdb_region": "us-east-1",
        }
    }

    with pytest.raises(ValidationError):
        validate_parametrization_file(missing_json)