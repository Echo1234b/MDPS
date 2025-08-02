import os
import json
import joblib
import pickle
import shutil
from datetime import datetime
import hashlib
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class ModelVersionControl:
    def __init__(self, repository_path: str = "./model_repository"):
        """
        Initialize the Model Version Control system.
        
        Args:
            repository_path: Path to the model repository
        """
        self.repository_path = repository_path
        self.models_dir = os.path.join(repository_path, "models")
        self.metadata_dir = os.path.join(repository_path, "metadata")
        self.index_file = os.path.join(repository_path, "index.json")
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Initialize index if it doesn't exist
        if not os.path.exists(self.index_file):
            self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the model index."""
        index = {
            "models": {},
            "current_version": None,
            "created_at": datetime.now().isoformat()
        }
        
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=4)
    
    def _load_index(self) -> Dict[str, Any]:
        """Load the model index."""
        with open(self.index_file, 'r') as f:
            return json.load(f)
    
    def _save_index(self, index: Dict[str, Any]):
        """Save the model index."""
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=4)
    
    def _generate_model_hash(self, model_path: str) -> str:
        """Generate a hash for the model file."""
        hash_md5 = hashlib.md5()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_next_version(self) -> str:
        """Get the next available version number."""
        index = self._load_index()
        versions = list(index["models"].keys())
        
        if not versions:
            return "v1.0.0"
        
        # Parse versions and find the next one
        latest_version = versions[-1]
        major, minor, patch = map(int, latest_version[1:].split('.'))
        
        # Increment patch version
        return f"v{major}.{minor}.{patch + 1}"
    
    def save_model(self, model: Any, model_name: str, version: str = None, 
                   metadata: Dict[str, Any] = None, tags: List[str] = None,
                   description: str = "") -> str:
        """
        Save a model to the repository.
        
        Args:
            model: Model object to save
            model_name: Name of the model
            version: Version of the model (if None, auto-generate)
            metadata: Additional metadata to save with the model
            tags: Tags to associate with the model
            description: Description of the model
            
        Returns:
            Version of the saved model
        """
        # Generate version if not provided
        if version is None:
            version = self._get_next_version()
        
        # Create model directory
        model_dir = os.path.join(self.models_dir, model_name, version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.pkl")
        
        # Try to save with joblib first, fall back to pickle
        try:
            joblib.dump(model, model_path)
        except:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Generate model hash
        model_hash = self._generate_model_hash(model_path)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        model_metadata = {
            "name": model_name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "model_hash": model_hash,
            "description": description,
            "tags": tags if tags else [],
            "metadata": metadata
        }
        
        # Save metadata
        metadata_path = os.path.join(self.metadata_dir, f"{model_name}_{version}.json")
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=4)
        
        # Update index
        index = self._load_index()
        
        if model_name not in index["models"]:
            index["models"][model_name] = {}
        
        index["models"][model_name][version] = {
            "metadata_path": metadata_path,
            "model_path": model_path,
            "created_at": datetime.now().isoformat()
        }
        
        # Set as current version
        index["current_version"] = f"{model_name}:{version}"
        
        self._save_index(index)
        
        return version
    
    def load_model(self, model_name: str, version: str = None) -> Any:
        """
        Load a model from the repository.
        
        Args:
            model_name: Name of the model
            version: Version of the model (if None, load latest)
            
        Returns:
            Loaded model object
        """
        index = self._load_index()
        
        # Get version
        if version is None:
            if model_name not in index["models"]:
                raise ValueError(f"Model '{model_name}' not found in repository")
            
            versions = list(index["models"][model_name].keys())
            if not versions:
                raise ValueError(f"No versions found for model '{model_name}'")
            
            version = versions[-1]
        
        # Get model path
        if model_name not in index["models"] or version not in index["models"][model_name]:
            raise ValueError(f"Model '{model_name}' version '{version}' not found in repository")
        
        model_path = index["models"][model_name][version]["model_path"]
        
        # Load model
        try:
            # Try to load with joblib first
            model = joblib.load(model_path)
        except:
            # Fall back to pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        return model
    
    def get_model_metadata(self, model_name: str, version: str = None) -> Dict[str, Any]:
        """
        Get metadata for a model.
        
        Args:
            model_name: Name of the model
            version: Version of the model (if None, get latest)
            
        Returns:
            Model metadata
        """
        index = self._load_index()
        
        # Get version
        if version is None:
            if model_name not in index["models"]:
                raise ValueError(f"Model '{model_name}' not found in repository")
            
            versions = list(index["models"][model_name].keys())
            if not versions:
                raise ValueError(f"No versions found for model '{model_name}'")
            
            version = versions[-1]
        
        # Get metadata path
        if model_name not in index["models"] or version not in index["models"][model_name]:
            raise ValueError(f"Model '{model_name}' version '{version}' not found in repository")
        
        metadata_path = index["models"][model_name][version]["metadata_path"]
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def list_models(self) -> pd.DataFrame:
        """
        List all models in the repository.
        
        Returns:
            DataFrame with model information
        """
        index = self._load_index()
        
        models_data = []
        
        for model_name, versions in index["models"].items():
            for version, info in versions.items():
                # Load metadata
                with open(info["metadata_path"], 'r') as f:
                    metadata = json.load(f)
                
                models_data.append({
                    "name": model_name,
                    "version": version,
                    "created_at": metadata["created_at"],
                    "description": metadata["description"],
                    "tags": ", ".join(metadata["tags"]),
                    "is_current": index["current_version"] == f"{model_name}:{version}"
                })
        
        return pd.DataFrame(models_data)
    
    def list_versions(self, model_name: str) -> pd.DataFrame:
        """
        List all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            DataFrame with version information
        """
        index = self._load_index()
        
        if model_name not in index["models"]:
            raise ValueError(f"Model '{model_name}' not found in repository")
        
        versions_data = []
        
        for version, info in index["models"][model_name].items():
            # Load metadata
            with open(info["metadata_path"], 'r') as f:
                metadata = json.load(f)
            
            versions_data.append({
                "version": version,
                "created_at": metadata["created_at"],
                "description": metadata["description"],
                "tags": ", ".join(metadata["tags"]),
                "is_current": index["current_version"] == f"{model_name}:{version}"
            })
        
        return pd.DataFrame(versions_data)
    
    def delete_model(self, model_name: str, version: str = None):
        """
        Delete a model or model version.
        
        Args:
            model_name: Name of the model
            version: Version of the model (if None, delete all versions)
        """
        index = self._load_index()
        
        if model_name not in index["models"]:
            raise ValueError(f"Model '{model_name}' not found in repository")
        
        if version is None:
            # Delete all versions
            for version, info in index["models"][model_name].items():
                # Delete model file
                if os.path.exists(info["model_path"]):
                    os.remove(info["model_path"])
                
                # Delete metadata file
                if os.path.exists(info["metadata_path"]):
                    os.remove(info["metadata_path"])
                
                # Delete model directory
                model_dir = os.path.dirname(info["model_path"])
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
            
            # Remove from index
            del index["models"][model_name]
            
            # Update current version if needed
            if index["current_version"] and index["current_version"].startswith(f"{model_name}:"):
                index["current_version"] = None
        else:
            # Delete specific version
            if version not in index["models"][model_name]:
                raise ValueError(f"Version '{version}' not found for model '{model_name}'")
            
            info = index["models"][model_name][version]
            
            # Delete model file
            if os.path.exists(info["model_path"]):
                os.remove(info["model_path"])
            
            # Delete metadata file
            if os.path.exists(info["metadata_path"]):
                os.remove(info["metadata_path"])
            
            # Delete model directory
            model_dir = os.path.dirname(info["model_path"])
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            
            # Remove from index
            del index["models"][model_name][version]
            
            # Update current version if needed
            if index["current_version"] == f"{model_name}:{version}":
                # Set to latest version if available
                versions = list(index["models"][model_name].keys())
                if versions:
                    index["current_version"] = f"{model_name}:{versions[-1]}"
                else:
                    index["current_version"] = None
        
        # Save updated index
        self._save_index(index)
    
    def set_current_version(self, model_name: str, version: str):
        """
        Set the current version of a model.
        
        Args:
            model_name: Name of the model
            version: Version to set as current
        """
        index = self._load_index()
        
        if model_name not in index["models"]:
            raise ValueError(f"Model '{model_name}' not found in repository")
        
        if version not in index["models"][model_name]:
            raise ValueError(f"Version '{version}' not found for model '{model_name}'")
        
        index["current_version"] = f"{model_name}:{version}"
        self._save_index(index)
    
    def get_current_version(self) -> Optional[str]:
        """
        Get the current model version.
        
        Returns:
            Current model version in format "model_name:version" or None if not set
        """
        index = self._load_index()
        return index["current_version"]
    
    def search_models(self, query: str = "", tags: List[str] = None) -> pd.DataFrame:
        """
        Search for models based on query and tags.
        
        Args:
            query: Search query for model name or description
            tags: List of tags to filter by
            
        Returns:
            DataFrame with matching models
        """
        models_df = self.list_models()
        
        if models_df.empty:
            return models_df
        
        # Filter by query
        if query:
            query = query.lower()
            mask = (
                models_df["name"].str.lower().str.contains(query) |
                models_df["description"].str.lower().str.contains(query)
            )
            models_df = models_df[mask]
        
        # Filter by tags
        if tags:
            tags_set = set(tags)
            mask = models_df["tags"].apply(
                lambda x: any(tag in x for tag in tags_set)
            )
            models_df = models_df[mask]
        
        return models_df
    
    def compare_versions(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two versions of a model.
        
        Args:
            model_name: Name of the model
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Dictionary with comparison results
        """
        # Get metadata for both versions
        metadata1 = self.get_model_metadata(model_name, version1)
        metadata2 = self.get_model_metadata(model_name, version2)
        
        # Compare basic information
        comparison = {
            "model_name": model_name,
            "version1": version1,
            "version2": version2,
            "created_at_diff": (
                datetime.fromisoformat(metadata2["created_at"]) - 
                datetime.fromisoformat(metadata1["created_at"])
            ).total_seconds(),
            "description_diff": metadata1["description"] != metadata2["description"],
            "tags_diff": set(metadata1["tags"]) != set(metadata2["tags"]),
            "metadata_diff": metadata1["metadata"] != metadata2["metadata"]
        }
        
        return comparison
    
    def export_model(self, model_name: str, version: str = None, export_path: str = None) -> str:
        """
        Export a model to a file.
        
        Args:
            model_name: Name of the model
            version: Version of the model (if None, export latest)
            export_path: Path to export to (if None, auto-generate)
            
        Returns:
            Path to the exported model
        """
        index = self._load_index()
        
        # Get version
        if version is None:
            if model_name not in index["models"]:
                raise ValueError(f"Model '{model_name}' not found in repository")
            
            versions = list(index["models"][model_name].keys())
            if not versions:
                raise ValueError(f"No versions found for model '{model_name}'")
            
            version = versions[-1]
        
        # Get model path
        if model_name not in index["models"] or version not in index["models"][model_name]:
            raise ValueError(f"Model '{model_name}' version '{version}' not found in repository")
        
        model_path = index["models"][model_name][version]["model_path"]
        
        # Generate export path if not provided
        if export_path is None:
            export_path = f"{model_name}_{version}.pkl"
        
        # Copy model file
        shutil.copy2(model_path, export_path)
        
        return export_path
    
    def import_model(self, model_path: str, model_name: str, version: str = None,
                     metadata: Dict[str, Any] = None, tags: List[str] = None,
                     description: str = "") -> str:
        """
        Import a model from a file.
        
        Args:
            model_path: Path to the model file
            model_name: Name to assign to the model
            version: Version to assign to the model (if None, auto-generate)
            metadata: Additional metadata to save with the model
            tags: Tags to associate with the model
            description: Description of the model
            
        Returns:
            Version of the imported model
        """
        # Load model
        try:
            # Try to load with joblib first
            model = joblib.load(model_path)
        except:
            # Fall back to pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        # Save model to repository
        return self.save_model(
            model=model,
            model_name=model_name,
            version=version,
            metadata=metadata,
            tags=tags,
            description=description
        )
