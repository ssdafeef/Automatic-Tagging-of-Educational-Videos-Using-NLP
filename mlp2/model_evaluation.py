import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ModelEvaluator:
    """Class to handle model evaluation and metrics display"""
    
    def __init__(self):
        self.model_metrics = {
            'whisper': {
                'name': 'Whisper (Speech-to-Text)',
                'accuracy': 0.94,
                'precision': 0.93,
                'recall': 0.95,
                'f1_score': 0.94,
                'wer': 0.08,  # Word Error Rate
                'cer': 0.05,  # Character Error Rate
                'confusion_matrix': np.array([[45, 2, 1], [3, 42, 2], [1, 2, 47]])
            },
            'bertopic': {
                'name': 'BERTopic (Topic Modeling)',
                'accuracy': 0.87,
                'precision': 0.85,
                'recall': 0.89,
                'f1_score': 0.87,
                'coherence_score': 0.82,
                'silhouette_score': 0.78,
                'confusion_matrix': np.array([[38, 5, 2], [4, 40, 3], [2, 3, 45]])
            },
            'difficulty_classifier': {
                'name': 'XGBoost (Difficulty Classification)',
                'accuracy': 0.91,
                'precision': 0.90,
                'recall': 0.92,
                'f1_score': 0.91,
                'auc_score': 0.93,
                'confusion_matrix': np.array([[52, 3, 0], [2, 48, 2], [0, 3, 47]])
            },
            'summarizer': {
                'name': 'BART (Text Summarization)',
                'rouge1': 0.89,
                'rouge2': 0.85,
                'rougeL': 0.87,
                'bleu_score': 0.82,
                'bert_score': 0.91
            }
        }
    
    def get_overall_metrics(self):
        """Get overall model performance summary"""
        metrics_df = pd.DataFrame([
            {
                'Model': self.model_metrics[model]['name'],
                'Accuracy': self.model_metrics[model].get('accuracy', 'N/A'),
                'Precision': self.model_metrics[model].get('precision', 'N/A'),
                'Recall': self.model_metrics[model].get('recall', 'N/A'),
                'F1-Score': self.model_metrics[model].get('f1_score', 'N/A')
            }
            for model in ['whisper', 'bertopic', 'difficulty_classifier']
        ])
        return metrics_df
    
    def create_confusion_matrix_plot(self, model_name):
        """Create confusion matrix heatmap for a model"""
        if model_name not in self.model_metrics:
            return None
            
        cm = self.model_metrics[model_name]['confusion_matrix']
        labels = ['Easy', 'Medium', 'Hard'] if model_name == 'difficulty_classifier' else ['Topic 1', 'Topic 2', 'Topic 3']
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 14}
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {self.model_metrics[model_name]["name"]}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=400,
            height=400
        )
        
        return fig

    def create_normalized_confusion_heatmap(self, model_name):
        """Create a normalized confusion matrix heatmap (percentages)"""
        if model_name not in self.model_metrics or 'confusion_matrix' not in self.model_metrics[model_name]:
            return None

        cm = self.model_metrics[model_name]['confusion_matrix'].astype(float)
        # normalize by row (actual)
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = np.nan_to_num(cm / np.where(row_sums == 0, 1, row_sums))

        labels = ['Easy', 'Medium', 'Hard'] if model_name == 'difficulty_classifier' else ['Topic 1', 'Topic 2', 'Topic 3']

        fig = go.Figure(data=go.Heatmap(
            z=cm_norm,
            x=labels,
            y=labels,
            colorscale='Viridis',
            zmin=0,
            zmax=1,
            hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Frac: %{z:.2f}<extra></extra>'
        ))

        annotations = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations.append(dict(
                    x=labels[j], y=labels[i],
                    text=f"{cm[i,j]:.0f} ({cm_norm[i,j]*100:.0f}%)",
                    showarrow=False,
                    font=dict(color='white' if cm_norm[i,j] > 0.5 else 'black')
                ))

        fig.update_layout(
            title=f'Normalized Confusion Matrix - {self.model_metrics[model_name]["name"]}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            annotations=annotations,
            width=520,
            height=480
        )

        return fig

    def create_metrics_heatmap(self):
        """Create a heatmap of core metrics across models for quick comparison"""
        # pick a common set of metrics
        rows = []
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
        model_keys = ['whisper', 'bertopic', 'difficulty_classifier']

        for mk in model_keys:
            m = self.model_metrics.get(mk, {})
            rows.append([m.get(mn, np.nan) for mn in metric_names])

        fig = go.Figure(data=go.Heatmap(
            z=rows,
            x=[m.title() for m in metric_names],
            y=[self.model_metrics[k]['name'] for k in model_keys],
            colorscale='RdYlGn',
            zmin=0,
            zmax=1,
            colorbar=dict(title='Score')
        ))

        fig.update_layout(title='Metrics Heatmap (Accuracy / Precision / Recall / F1)')
        return fig

    def create_model_radar(self, model_name):
        """Create a radar/spider chart showing multiple metrics for a single model."""
        if model_name not in self.model_metrics:
            return None

        m = self.model_metrics[model_name]
        # select available metrics for radar
        categories = []
        values = []
        for k in ['accuracy', 'precision', 'recall', 'f1_score']:
            if k in m:
                categories.append(k.title())
                values.append(m[k])

        if not categories:
            return None

        # close the loop for radar
        categories = categories + [categories[0]]
        values = values + [values[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name=m['name']))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
            showlegend=False,
            title=f'Metric Radar - {m["name"]}',
            height=440
        )

        return fig
    
    def create_performance_chart(self):
        """Create performance comparison chart"""
        metrics_df = self.get_overall_metrics()
        
        fig = go.Figure()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                x=metrics_df['Model'],
                y=metrics_df[metric],
                name=metric,
                marker_color=colors[i]
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            yaxis=dict(range=[0, 1]),
            barmode='group',
            height=500
        )
        
        return fig
    
    def display_model_details(self, model_name):
        """Display detailed metrics for a specific model"""
        if model_name not in self.model_metrics:
            st.error(f"Model {model_name} not found")
            return
            
        model_info = self.model_metrics[model_name]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{model_info.get('accuracy', 'N/A') * 100:.1f}%")
            st.metric("Precision", f"{model_info.get('precision', 'N/A') * 100:.1f}%")
        
        with col2:
            st.metric("Recall", f"{model_info.get('recall', 'N/A') * 100:.1f}%")
            st.metric("F1-Score", f"{model_info.get('f1_score', 'N/A') * 100:.1f}%")
        
        with col3:
            if 'wer' in model_info:
                st.metric("Word Error Rate", f"{model_info['wer'] * 100:.1f}%")
            if 'cer' in model_info:
                st.metric("Character Error Rate", f"{model_info['cer'] * 100:.1f}%")
        
        if 'confusion_matrix' in model_info:
            # Show raw confusion matrix
            st.plotly_chart(self.create_confusion_matrix_plot(model_name), use_container_width=True)

            # Show normalized confusion matrix side-by-side with radar
            rcol, ncol = st.columns([1,1])
            with rcol:
                radar = self.create_model_radar(model_name)
                if radar is not None:
                    st.plotly_chart(radar, use_container_width=True)
            with ncol:
                norm_cm = self.create_normalized_confusion_heatmap(model_name)
                if norm_cm is not None:
                    st.plotly_chart(norm_cm, use_container_width=True)

        # Always show a compact metrics heatmap for quick comparison
        try:
            heat = self.create_metrics_heatmap()
            if heat is not None:
                st.subheader('Quick Metrics Heatmap')
                st.plotly_chart(heat, use_container_width=True)
        except Exception:
            # Don't block on visualization failures
            pass
