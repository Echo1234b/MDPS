import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import accuracy_score, mean_squared_error
import shap
import lime
import lime.lime_tabular
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    def __init__(self, model, task_type='classification', feature_names=None, 
                 class_names=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the Model Explainer.
        
        Args:
            model: Model to explain
            task_type: 'classification' or 'regression'
            feature_names: List of feature names
            class_names: List of class names (for classification)
            device: Device to use for PyTorch models
        """
        self.model = model
        self.task_type = task_type
        self.feature_names = feature_names
        self.class_names = class_names
        self.device = device
        
        # Store explanations
        self.explanations = {}
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
    
    def explain_with_shap(self, X, background_data=None, nsamples=100, plot_type='summary'):
        """
        Explain model predictions using SHAP.
        
        Args:
            X: Feature data to explain
            background_data: Background data for SHAP (if None, use X)
            nsamples: Number of samples for SHAP explanations
            plot_type: Type of SHAP plot ('summary', 'dependence', 'force', 'waterfall')
            
        Returns:
            SHAP values
        """
        # Use X as background data if not provided
        if background_data is None:
            background_data = X
        
        # Initialize SHAP explainer based on model type
        if self.shap_explainer is None:
            if hasattr(self.model, 'predict_proba'):
                self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, background_data)
            else:
                self.shap_explainer = shap.KernelExplainer(self.model.predict, background_data)
        
        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(X, nsamples=nsamples)
        
        # Store explanations
        self.explanations['shap_values'] = shap_values
        
        # Plot explanations
        if plot_type == 'summary':
            self._plot_shap_summary(shap_values, X)
        elif plot_type == 'dependence':
            self._plot_shap_dependence(shap_values, X)
        elif plot_type == 'force':
            self._plot_shap_force(shap_values, X)
        elif plot_type == 'waterfall':
            self._plot_shap_waterfall(shap_values, X)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        return shap_values
    
    def _plot_shap_summary(self, shap_values, X):
        """Plot SHAP summary plot."""
        plt.figure(figsize=(10, 6))
        
        if self.task_type == 'classification' and isinstance(shap_values, list):
            # Multi-class classification
            for i, class_shap_values in enumerate(shap_values):
                class_name = self.class_names[i] if self.class_names else f"Class {i}"
                shap.summary_plot(
                    class_shap_values, 
                    X, 
                    feature_names=self.feature_names,
                    show=False,
                    plot_type="bar"
                )
                plt.title(f"SHAP Summary - {class_name}")
                plt.tight_layout()
                plt.show()
        else:
            # Binary classification or regression
            shap.summary_plot(
                shap_values, 
                X, 
                feature_names=self.feature_names,
                show=False
            )
            plt.title("SHAP Summary")
            plt.tight_layout()
            plt.show()
    
    def _plot_shap_dependence(self, shap_values, X, feature_idx=0):
        """Plot SHAP dependence plot."""
        plt.figure(figsize=(10, 6))
        
        if self.task_type == 'classification' and isinstance(shap_values, list):
            # Multi-class classification
            for i, class_shap_values in enumerate(shap_values):
                class_name = self.class_names[i] if self.class_names else f"Class {i}"
                shap.dependence_plot(
                    feature_idx, 
                    class_shap_values, 
                    X, 
                    feature_names=self.feature_names,
                    show=False
                )
                plt.title(f"SHAP Dependence - {class_name}")
                plt.tight_layout()
                plt.show()
        else:
            # Binary classification or regression
            shap.dependence_plot(
                feature_idx, 
                shap_values, 
                X, 
                feature_names=self.feature_names,
                show=False
            )
            plt.title("SHAP Dependence")
            plt.tight_layout()
            plt.show()
    
    def _plot_shap_force(self, shap_values, X, sample_idx=0):
        """Plot SHAP force plot."""
        if self.task_type == 'classification' and isinstance(shap_values, list):
            # Multi-class classification
            for i, class_shap_values in enumerate(shap_values):
                class_name = self.class_names[i] if self.class_names else f"Class {i}"
                shap.force_plot(
                    self.shap_explainer.expected_value[i], 
                    class_shap_values[sample_idx], 
                    X[sample_idx], 
                    feature_names=self.feature_names,
                    matplotlib=True,
                    show=False
                )
                plt.title(f"SHAP Force - {class_name}")
                plt.tight_layout()
                plt.show()
        else:
            # Binary classification or regression
            expected_value = self.shap_explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[0]
            
            shap.force_plot(
                expected_value, 
                shap_values[sample_idx], 
                X[sample_idx], 
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            plt.title("SHAP Force")
            plt.tight_layout()
            plt.show()
    
    def _plot_shap_waterfall(self, shap_values, X, sample_idx=0):
        """Plot SHAP waterfall plot."""
        if self.task_type == 'classification' and isinstance(shap_values, list):
            # Multi-class classification
            for i, class_shap_values in enumerate(shap_values):
                class_name = self.class_names[i] if self.class_names else f"Class {i}"
                shap.waterfall_plot(
                    shap.Explanation(
                        values=class_shap_values[sample_idx], 
                        base_values=self.shap_explainer.expected_value[i], 
                        data=X[sample_idx], 
                        feature_names=self.feature_names
                    ),
                    show=False
                )
                plt.title(f"SHAP Waterfall - {class_name}")
                plt.tight_layout()
                plt.show()
        else:
            # Binary classification or regression
            expected_value = self.shap_explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[0]
            
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[sample_idx], 
                    base_values=expected_value, 
                    data=X[sample_idx], 
                    feature_names=self.feature_names
                ),
                show=False
            )
            plt.title("SHAP Waterfall")
            plt.tight_layout()
            plt.show()
    
    def explain_with_lime(self, X, sample_idx=0, num_features=10, num_samples=5000):
        """
        Explain model predictions using LIME.
        
        Args:
            X: Feature data
            sample_idx: Index of the sample to explain
            num_features: Number of features to include in explanation
            num_samples: Number of samples for LIME to generate
            
        Returns:
            LIME explanation
        """
        # Initialize LIME explainer
        if self.lime_explainer is None:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X, 
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='classification' if self.task_type == 'classification' else 'regression'
            )
        
        # Get sample to explain
        sample = X[sample_idx]
        
        # Generate explanation
        if self.task_type == 'classification':
            explanation = self.lime_explainer.explain_instance(
                sample, 
                self.model.predict_proba, 
                num_features=num_features, 
                num_samples=num_samples
            )
        else:
            explanation = self.lime_explainer.explain_instance(
                sample, 
                self.model.predict, 
                num_features=num_features, 
                num_samples=num_samples
            )
        
        # Store explanation
        self.explanations['lime'] = explanation
        
        # Plot explanation
        explanation.as_pyplot_figure()
        plt.tight_layout()
        plt.show()
        
        return explanation
    
    def explain_with_permutation_importance(self, X, y, n_repeats=10, random_state=42):
        """
        Explain model using permutation importance.
        
        Args:
            X: Feature data
            y: Target data
            n_repeats: Number of times to permute each feature
            random_state: Random seed
            
        Returns:
            Permutation importance
        """
        # Calculate permutation importance
        result = permutation_importance(
            self.model, X, y, n_repeats=n_repeats, random_state=random_state
        )
        
        # Store results
        self.explanations['permutation_importance'] = result
        
        # Plot results
        self._plot_permutation_importance(result)
        
        return result
    
    def _plot_permutation_importance(self, result):
        """Plot permutation importance."""
        plt.figure(figsize=(10, 6))
        
        sorted_idx = result.importances_mean.argsort()
        
        plt.boxplot(
            result.importances[sorted_idx].T,
            vert=False,
            labels=np.array(self.feature_names)[sorted_idx] if self.feature_names else sorted_idx
        )
        
        plt.title("Permutation Importance")
        plt.tight_layout()
        plt.show()
    
    def explain_with_partial_dependence(self, X, features, grid_resolution=100):
        """
        Explain model using partial dependence plots.
        
        Args:
            X: Feature data
            features: Features to plot (indices or names)
            grid_resolution: Number of points in the grid
            
        Returns:
            Partial dependence results
        """
        # Convert feature names to indices if needed
        if self.feature_names is not None:
            feature_indices = []
            for feature in features:
                if isinstance(feature, str):
                    if feature in self.feature_names:
                        feature_indices.append(self.feature_names.index(feature))
                    else:
                        raise ValueError(f"Feature '{feature}' not found in feature names.")
                else:
                    feature_indices.append(feature)
        else:
            feature_indices = features
        
        # Calculate partial dependence
        results = []
        for feature_idx in feature_indices:
            pd_results = partial_dependence(
                self.model, X, features=[feature_idx], grid_resolution=grid_resolution
            )
            results.append(pd_results)
        
        # Store results
        self.explanations['partial_dependence'] = results
        
        # Plot results
        self._plot_partial_dependence(results, feature_indices)
        
        return results
    
    def _plot_partial_dependence(self, results, feature_indices):
        """Plot partial dependence."""
        fig, axes = plt.subplots(1, len(feature_indices), figsize=(5 * len(feature_indices), 5))
        if len(feature_indices) == 1:
            axes = [axes]
        
        for i, (result, feature_idx) in enumerate(zip(results, feature_indices)):
            ax = axes[i]
            
            # Get feature name
            feature_name = self.feature_names[feature_idx] if self.feature_names else f"Feature {feature_idx}"
            
            # Plot partial dependence
            ax.plot(result['values'][0], result['average'][0])
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Partial Dependence')
            ax.set_title(f'Partial Dependence for {feature_name}')
        
        plt.tight_layout()
        plt.show()
    
    def explain_pytorch_model(self, X, sample_idx=0, method='integrated_gradients'):
        """
        Explain PyTorch model using various methods.
        
        Args:
            X: Feature data
            sample_idx: Index of the sample to explain
            method: Explanation method ('integrated_gradients', 'saliency', 'input_x_gradient')
            
        Returns:
            Explanation
        """
        if not isinstance(self.model, nn.Module):
            raise ValueError("Model must be a PyTorch nn.Module for this method.")
        
        # Set model to eval mode
        self.model.eval()
        
        # Get sample to explain
        sample = torch.FloatTensor(X[sample_idx]).to(self.device)
        sample.requires_grad = True
        
        # Forward pass
        output = self.model(sample.unsqueeze(0))
        
        # Get prediction
        if self.task_type == 'classification':
            pred_class = output.argmax(dim=1).item()
            pred_score = output[0, pred_class].item()
        else:
            pred_score = output.item()
        
        # Calculate explanation based on method
        if method == 'integrated_gradients':
            explanation = self._integrated_gradients(X, sample_idx)
        elif method == 'saliency':
            explanation = self._saliency_map(sample)
        elif method == 'input_x_gradient':
            explanation = self._input_x_gradient(sample)
        else:
            raise ValueError(f"Unknown explanation method: {method}")
        
        # Store explanation
        self.explanations[method] = explanation
        
        # Plot explanation
        self._plot_pytorch_explanation(explanation, sample_idx, method)
        
        return explanation
    
    def _integrated_gradients(self, X, sample_idx, baseline=None, num_steps=50):
        """Calculate integrated gradients."""
        # Get sample to explain
        sample = torch.FloatTensor(X[sample_idx]).to(self.device)
        
        # Use zero baseline if not provided
        if baseline is None:
            baseline = torch.zeros_like(sample)
        
        # Generate interpolated samples
        alphas = torch.linspace(0, 1, num_steps).to(self.device)
        alphas = alphas.view(num_steps, 1)
        
        # Calculate interpolated samples
        interpolated_samples = baseline + alphas * (sample - baseline)
        
        # Calculate gradients for all interpolated samples
        gradients = []
        for interpolated_sample in interpolated_samples:
            interpolated_sample = interpolated_sample.unsqueeze(0)
            interpolated_sample.requires_grad = True
            
            # Forward pass
            output = self.model(interpolated_sample)
            
            # Get prediction score
            if self.task_type == 'classification':
                pred_class = output.argmax(dim=1).item()
                score = output[0, pred_class]
            else:
                score = output
            
            # Backward pass
            self.model.zero_grad()
            score.backward()
            
            # Store gradients
            gradients.append(interpolated_sample.grad.detach().cpu().numpy())
        
        # Convert to numpy array
        gradients = np.array(gradients)
        
        # Calculate integrated gradients
        integrated_gradients = np.mean(gradients, axis=0) * (sample.detach().cpu().numpy() - baseline.detach().cpu().numpy())
        
        return integrated_gradients
    
    def _saliency_map(self, sample):
        """Calculate saliency map."""
        # Forward pass
        output = self.model(sample.unsqueeze(0))
        
        # Get prediction score
        if self.task_type == 'classification':
            pred_class = output.argmax(dim=1).item()
            score = output[0, pred_class]
        else:
            score = output
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Get gradients
        saliency = sample.grad.detach().cpu().numpy()
        
        return saliency
    
    def _input_x_gradient(self, sample):
        """Calculate input * gradient."""
        # Forward pass
        output = self.model(sample.unsqueeze(0))
        
        # Get prediction score
        if self.task_type == 'classification':
            pred_class = output.argmax(dim=1).item()
            score = output[0, pred_class]
        else:
            score = output
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Get gradients
        gradients = sample.grad.detach().cpu().numpy()
        
        # Calculate input * gradient
        input_x_gradient = sample.detach().cpu().numpy() * gradients
        
        return input_x_gradient
    
    def _plot_pytorch_explanation(self, explanation, sample_idx, method):
        """Plot PyTorch model explanation."""
        plt.figure(figsize=(10, 6))
        
        # Get feature names
        feature_names = self.feature_names if self.feature_names else [f"Feature {i}" for i in range(len(explanation))]
        
        # Create bar plot
        plt.barh(feature_names, explanation)
        
        # Add labels and title
        plt.xlabel('Attribution')
        plt.ylabel('Feature')
        plt.title(f'{method.replace("_", " ").title()} - Sample {sample_idx}')
        
        plt.tight_layout()
        plt.show()
    
    def save_explainer(self, filepath):
        """Save the explainer to a file."""
        # Create a dictionary of all attributes
        explainer_data = {
            'task_type': self.task_type,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'explanations': self.explanations
        }
        
        # Save to file
        joblib.dump(explainer_data, filepath)
        
        # Save SHAP explainer if available
        if self.shap_explainer is not None:
            joblib.dump(self.shap_explainer, f"{filepath}_shap.pkl")
        
        # Save LIME explainer if available
        if self.lime_explainer is not None:
            with open(f"{filepath}_lime.pkl", 'wb') as f:
                import pickle
                pickle.dump(self.lime_explainer, f)
    
    def load_explainer(self, filepath):
        """Load an explainer from a file."""
        # Load from file
        explainer_data = joblib.load(filepath)
        
        # Set attributes
        self.task_type = explainer_data['task_type']
        self.feature_names = explainer_data['feature_names']
        self.class_names = explainer_data['class_names']
        self.explanations = explainer_data['explanations']
        
        # Load SHAP explainer if available
        try:
            self.shap_explainer = joblib.load(f"{filepath}_shap.pkl")
        except:
            pass
        
        # Load LIME explainer if available
        try:
            with open(f"{filepath}_lime.pkl", 'rb') as f:
                import pickle
                self.lime_explainer = pickle.load(f)
        except:
            pass
        
        return self
