"""
This is a class for converting parameter data types from json text format and vice-versa.
MLFlow logs parameters as strings for efficiency and readability purposes. This class can use
a saved parameter schema to cast MLFlow params to their correct types. 
"""
from typing import Any
import json 

class ParamTypeCaster:
    def __init__(self, schema: dict[str, Any]=None):
        self.schema = schema or {}

        # Mapping from string type to casting function
        self.str_to_type = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool
        }

        # Inverse mapping for saving to json file
        self.type_to_str = {value: key for key, value in self.str_to_type.items()}

    def cast_params(self, params: dict[str, str]) -> dict[str, Any]:
        if not self.schema:
            raise ValueError("No schema provided")
        
        return {
            k: self.schema[k](v) if k in self.schema else v
            for k, v in params.items()
        }
    
    def save_schema(self, file_path: str) -> None:
        serializable = {
            k: self.type_to_str.get(v, "str")  
            for k, v in self.schema.items()
        }
        with open(file_path, "w") as f:
            json.dump(serializable, f, indent=2)

    def load_schema(self, file_path) -> None:
        with open(file_path, "r") as f:
            loaded = json.load(f)

        self.schema = {
            k: self.str_to_type.get(v, str)  # default to str if unknown
            for k, v in loaded.items()
        }
