import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
import time
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PerformanceTracker:
    def __init__(self, model_name, task_type='classification', metrics=None, 
                 save_dir='./performance_logs', track_time=True):
        """
        Initialize the Performance Tracker.
        
        Args:
            model_name: Name of the model being tracked
            task_type: 'classification' or 'regression'
            metrics: List of metrics to track (if None, use default metrics)
            save_dir: Directory to save performance logs
            track_time: Whether to track training and prediction time
        """
        self.model_name = model_name
        self.task_type = task_type
        self.save_dir = save_dir
        self.track_time = track_time
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Set default metrics based on task type
        if metrics is None:
            if task_type == 'classification':
                self.metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            else:
                self.metrics = ['mse', 'mae', 'r2', 'rmse']
        else:
            self.metrics = metrics
        
        # Initialize performance history
        self.performance_history = []
        self.training_time = []
        self.prediction_time = []
        
        # Initialize learning curve data
        self.learning_curve_data = None
    
    def log_performance(self, y_true, y_pred, y_proba=None, dataset_name='validation', 
                      iteration=None, additional_metrics=None):
        """
        Log performance metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            y_proba: Predicted probabilities (for classification)
            dataset_name: Name of the dataset (train, validation, test)
            iteration: Iteration number (e.g., epoch)
            additional_metrics: Dictionary of additional metrics to log
            
        Returns:
            Dictionary of performance metrics
        """
        # Calculate metrics
        metrics_dict = {}
        
        if self.task_type == 'classification':
            # Classification metrics
            metrics_dict['accuracy'] = accuracy_score(y_true, y_pred)
            metrics_dict['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics_dict['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics_dict['f1'] = f1_score(y_true, y_pred, average='weighted')
            
            # ROC AUC if probabilities are provided
            if y_proba is not None:
                try:
                    if len(np.unique(y_true)) == 2:
                        # Binary classification
                        metrics_dict['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                    else:
                        # Multi-class classification
                        metrics_dict['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                except Exception as e:
                    print(f"Warning: Could not calculate ROC AUC: {str(e)}")
        else:
            # Regression metrics
            metrics_dict['mse'] = mean_squared_error(y_true, y_pred)
            metrics_dict['mae'] = mean_absolute_error(y_true, y_pred)
            metrics_dict['r2'] = r2_score(y_true, y_pred)
            metrics_dict['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Add additional metrics if provided
        if additional_metrics is not None:
            metrics_dict.update(additional_metrics)
        
        # Create log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'dataset': dataset_name,
            'iteration': iteration,
            'metrics': metrics_dict
        }
        
        # Add to performance history
        self.performance_history.append(log_entry)
        
        # Save to file
        self._save_log_entry(log_entry)
        
        return metrics_dict
    
    def log_training_time(self, time_seconds, iteration=None):
        """
        Log training time.
        
        Args:
            time_seconds: Training time in seconds
            iteration: Iteration number
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'type': 'training_time',
            'value': time_seconds,
            'iteration': iteration
        }
        
        self.training_time.append(log_entry)
        self._save_log_entry(log_entry)
    
    def log_prediction_time(self, time_seconds, num_samples, iteration=None):
        """
        Log prediction time.
        
        Args:
            time_seconds: Prediction time in seconds
            num_samples: Number of samples predicted
            iteration: Iteration number
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'type': 'prediction_time',
            'value': time_seconds,
            'num_samples': num_samples,
            'time_per_sample': time_seconds / num_samples,
            'iteration': iteration
        }
        
        self.prediction_time.append(log_entry)
        self._save_log_entry(log_entry)
    
    def log_learning_curve(self, model, X, y, cv=5, train_sizes=None, scoring=None):
        """
        Log learning curve data.
        
        Args:
            model: Model to evaluate
            X: Feature data
            y: Target data
            cv: Number of cross-validation folds
            train_sizes: Training set sizes for learning curve
            scoring: Scoring metric
        """
        # Set default scoring based on task type
        if scoring is None:
            scoring = 'accuracy' if self.task_type == 'classification' else 'neg_mean_squared_error'
        
        # Calculate learning curve
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring
        )
        
        # Store learning curve data
        self.learning_curve_data = {
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': train_scores.mean(axis=1).tolist(),
            'train_scores_std': train_scores.std(axis=1).tolist(),
            'val_scores_mean': val_scores.mean(axis=1).tolist(),
            'val_scores_std': val_scores.std(axis=1).tolist(),
            'scoring': scoring
        }
        
        # Save to file
        with open(os.path.join(self.save_dir, f"{self.model_name}_learning_curve.json"), 'w') as f:
            json.dump(self.learning_curve_data, f)
    
    def plot_performance_history(self, metric=None, dataset='validation', figsize=(10, 6)):
        """
        Plot performance history.
        
        Args:
            metric: Metric to plot (if None, plot all metrics)
            dataset: Dataset to plot (train, validation, test)
            figsize: Figure size
        """
        # Filter performance history by dataset
        history = [entry for entry in self.performance_history if entry['dataset'] == dataset]
        
        if not history:
            print(f"No performance history found for dataset: {dataset}")
            return
        
        # Create DataFrame from performance history
        df = pd.DataFrame([
            {
                'iteration': entry['iteration'],
                **entry['metrics']
            }
            for entry in history
        ])
        
        # Plot metrics
        if metric is None:
            # Plot all metrics
            metrics_to_plot = [m for m in self.metrics if m in df.columns]
            n_metrics = len(metrics_to_plot)
            
            if n_metrics == 0:
                print("No metrics to plot.")
                return
            
            fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, squeeze=False)
            axes = axes.flatten()
            
            for i, metric_name in enumerate(metrics_to_plot):
                axes[i].plot(df['iteration'], df[metric_name], marker='o')
                axes[i].set_title(f'{metric_name.capitalize()} - {dataset.capitalize()}')
                axes[i].set_xlabel('Iteration')
                axes[i].set_ylabel(metric_name.capitalize())
                axes[i].grid(True)
            
            plt.tight_layout()
            plt.show()
        else:
            # Plot specific metric
            if metric not in df.columns:
                print(f"Metric '{metric}' not found in performance history.")
                return
            
            plt.figure(figsize=figsize)
            plt.plot(df['iteration'], df[metric], marker='o')
            plt.title(f'{metric.capitalize()} - {dataset.capitalize()}')
            plt.xlabel('Iteration')
            plt.ylabel(metric.capitalize())
            plt.grid(True)
            plt.show()
    
    def plot_learning_curve(self, figsize=(10, 6)):
        """
        Plot learning curve.
        
        Args:
            figsize: Figure size
        """
        if self.learning_curve_data is None:
            print("No learning curve data found.")
            return
        
        # Extract data
        train_sizes = self.learning_curve_data['train_sizes']
        train_scores_mean = self.learning_curve_data['train_scores_mean']
        train_scores_std = self.learning_curve_data['train_scores_std']
        val_scores_mean = self.learning_curve_data['val_scores_mean']
        val_scores_std = self.learning_curve_data['val_scores_std']
        scoring = self.learning_curve_data['scoring']
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot training scores
        plt.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="Training score")
        plt.fill_between(
            train_sizes,
            np.array(train_scores_mean) - np.array(train_scores_std),
            np.array(train_scores_mean) + np.array(train_scores_std),
            alpha=0.1, color="blue"
        )
        
        # Plot validation scores
        plt.plot(train_sizes, val_scores_mean, 'o-', color="red", label="Cross-validation score")
        plt.fill_between(
            train_sizes,
            np.array(val_scores_mean) - np.array(val_scores_std),
            np.array(val_scores_mean) + np.array(val_scores_std),
            alpha=0.1, color="red"
        )
        
        # Add labels and title
        plt.title(f"Learning Curve - {self.model_name}")
        plt.xlabel("Training examples")
        plt.ylabel(f"Score ({scoring})")
        plt.legend(loc="best")
        plt.grid(True)
        
        plt.show()
    
    def plot_time_metrics(self, figsize=(12, 6)):
        """
        Plot time metrics.
        
        Args:
            figsize: Figure size
        """
        if not self.track_time:
            print("Time tracking is disabled.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot training time
        if self.training_time:
            df_train = pd.DataFrame(self.training_time)
            axes[0].plot(df_train['iteration'], df_train['value'], marker='o')
            axes[0].set_title('Training Time')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('Time (seconds)')
            axes[0].grid(True)
        else:
            axes[0].text(0.5, 0.5, 'No training time data', ha='center', va='center')
            axes[0].set_title('Training Time')
        
        # Plot prediction time
        if self.prediction_time:
            df_pred = pd.DataFrame(self.prediction_time)
            axes[1].plot(df_pred['iteration'], df_pred['time_per_sample'], marker='o')
            axes[1].set_title('Prediction Time per Sample')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Time (seconds)')
            axes[1].grid(True)
        else:
            axes[1].text(0.5, 0.5, 'No prediction time data', ha='center', va='center')
            axes[1].set_title('Prediction Time per Sample')
        
        plt.tight_layout()
        plt.show()
    
    def get_performance_summary(self, dataset='validation', metric=None):
        """
        Get performance summary.
        
        Args:
            dataset: Dataset to summarize (train, validation, test)
            metric: Metric to summarize (if None, summarize all metrics)
            
        Returns:
            Performance summary
        """
        # Filter performance history by dataset
        history = [entry for entry in self.performance_history if entry['dataset'] == dataset]
        
        if not history:
            print(f"No performance history found for dataset: {dataset}")
            return None
        
        # Create DataFrame from performance history
        df = pd.DataFrame([
            {
                'iteration': entry['iteration'],
                **entry['metrics']
            }
            for entry in history
        ])
        
        # Get summary statistics
        if metric is None:
            # Summarize all metrics
            metrics_to_summarize = [m for m in self.metrics if m in df.columns]
            summary = df[metrics_to_summarize].describe()
        else:
            # Summarize specific metric
            if metric not in df.columns:
                print(f"Metric '{metric}' not found in performance history.")
                return None
            
            summary = df[metric].describe()
        
        return summary
    
    def compare_models(self, other_trackers, metric=None, dataset='validation', figsize=(10, 6)):
        """
        Compare performance with other models.
        
        Args:
            other_trackers: List of other PerformanceTracker objects
            metric: Metric to compare (if None, compare all metrics)
            dataset: Dataset to compare (train, validation, test)
            figsize: Figure size
        """
        # Get all trackers including this one
        all_trackers = [self] + other_trackers
        
        # Create DataFrame with all performance data
        all_data = []
        for tracker in all_trackers:
            history = [entry for entry in tracker.performance_history if entry['dataset'] == dataset]
            
            for entry in history:
                for metric_name, value in entry['metrics'].items():
                    all_data.append({
                        'model': tracker.model_name,
                        'iteration': entry['iteration'],
                        'metric': metric_name,
                        'value': value
                    })
        
        if not all_data:
            print(f"No performance data found for dataset: {dataset}")
            return
        
        df = pd.DataFrame(all_data)
        
        # Plot comparison
        if metric is None:
            # Plot all metrics
            metrics_to_plot = [m for m in self.metrics if m in df['metric'].unique()]
            n_metrics = len(metrics_to_plot)
            
            if n_metrics == 0:
                print("No metrics to plot.")
                return
            
            fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, squeeze=False)
            axes = axes.flatten()
            
            for i, metric_name in enumerate(metrics_to_plot):
                metric_data = df[df['metric'] == metric_name]
                
                for model in df['model'].unique():
                    model_data = metric_data[metric_data['model'] == model]
                    axes[i].plot(model_data['iteration'], model_data['value'], marker='o', label=model)
                
                axes[i].set_title(f'{metric_name.capitalize()} - {dataset.capitalize()}')
                axes[i].set_xlabel('Iteration')
                axes[i].set_ylabel(metric_name.capitalize())
                axes[i].legend()
                axes[i].grid(True)
            
            plt.tight_layout()
            plt.show()
        else:
            # Plot specific metric
            if metric not in df['metric'].unique():
                print(f"Metric '{metric}' not found in performance data.")
                return
            
            metric_data = df[df['metric'] == metric]
            
            plt.figure(figsize=figsize)
            
            for model in df['model'].unique():
                model_data = metric_data[metric_data['model'] == model]
                plt.plot(model_data['iteration'], model_data['value'], marker='o', label=model)
            
            plt.title(f'{metric.capitalize()} - {dataset.capitalize()}')
            plt.xlabel('Iteration')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            plt.show()
    
    def _save_log_entry(self, log_entry):
        """Save a log entry to a file."""
        # Create filename based on model name and entry type
        if 'metrics' in log_entry:
            filename = f"{self.model_name}_performance.json"
        elif log_entry['type'] == 'training_time':
            filename = f"{self.model_name}_training_time.json"
        elif log_entry['type'] == 'prediction_time':
            filename = f"{self.model_name}_prediction_time.json"
        else:
            filename = f"{self.model_name}_logs.json"
        
        filepath = os.path.join(self.save_dir, filename)
        
        # Load existing data if file exists
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            data = []
        
        # Add new entry
        data.append(log_entry)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def save_tracker(self, filepath=None):
        """
        Save the tracker to a file.
        
        Args:
            filepath: Path to save the tracker (if None, use default path)
        """
        if filepath is None:
            filepath = os.path.join(self.save_dir, f"{self.model_name}_tracker.pkl")
        
        # Create a dictionary of all attributes
        tracker_data = {
            'model_name': self.model_name,
            'task_type': self.task_type,
            'metrics': self.metrics,
            'performance_history': self.performance_history,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'learning_curve_data': self.learning_curve_data
        }
        
        # Save to file
        import joblib
        joblib.dump(tracker_data, filepath)
    
    def load_tracker(self, filepath):
        """
        Load a tracker from a file.
        
        Args:
            filepath: Path to the tracker file
            
        Returns:
            Loaded PerformanceTracker object
        """
        # Load from file
        import joblib
        tracker_data = joblib.load(filepath)
        
        # Set attributes
        self.model_name = tracker_data['model_name']
        self.task_type = tracker_data['task_type']
        self.metrics = tracker_data['metrics']
        self.performance_history = tracker_data['performance_history']
        self.training_time = tracker_data['training_time']
        self.prediction_time = tracker_data['prediction_time']
        self.learning_curve_data = tracker_data['learning_curve_data']
        
        return self
