#!/usr/bin/env python3
"""
Smart Contract Vulnerability Detection using Wide + TabTransformer Neural Network

Enhanced version with improved training stability and performance optimizations
"""

import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path

from IPython.display import display, Image
import re

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Local imports
from config.enhanced_fragment_vectorizer import EnhancedFragmentVectorizer
from config.models.wide_tabtransformer import WideTabTransformer
from config.arg_parser import parameter_parser

# For K-fold cross validation
from sklearn.model_selection import StratifiedKFold

# Configuration
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Configure matplotlib for Colab
import matplotlib
matplotlib.use('Agg')
plt.ioff()

# Global configuration
CONFIG = {
    'MIN_FRAGMENT_LENGTH': 3,
    'MAX_FRAGMENT_LENGTH': 500,
    'CACHE_DIR': 'cache',
    'RESULTS_DIR': 'results',
    'PLOTS_DIR': 'plots',
    'MODELS_DIR': 'models'
}

def setup_directories():
    """Create necessary directories"""
    for dir_name in CONFIG.values():
        if isinstance(dir_name, str) and dir_name not in ['', '.']:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            
def print_header():
    """Print application header"""
    print("=" * 80)
    print("SMART CONTRACT VULNERABILITY DETECTION")
    print("Wide + TabTransformer Neural Network")
    print("Enhanced Vectorization System v2.0")
    print("=" * 80)

def print_parameters(args):
    """Print all parameters in organized sections"""
    print("\n" + "="*60)
    print("CONFIGURATION PARAMETERS")
    print("="*60)
    
    sections = {
        "Data": ['filename', 'vt', 'use_enhanced_vectorization', 'use_data_augmentation', 'use_smote'],
        "Architecture": ['wide_features', 'num_transformer_layers', 'num_heads', 'embedding_dim', 'mlp_hidden_dim'],
        "Training": ['lr', 'epochs', 'batch_size', 'dropout', 'early_stopping_patience', 'reduce_lr_patience'],
        "Regularization": ['l1_reg', 'l2_reg', 'gradient_clip'],
        "Vectorization": ['vec_length', 'w2v_window', 'w2v_min_count', 'w2v_epochs', 'w2v_negative'],
        "Advanced": ['progressive_training', 'use_kfold', 'kfold_splits', 'tensorboard', 'save_best_model']
    }
    
    for section, params in sections.items():
        print(f"\n{section} Parameters:")
        print("-" * 40)
        for param in params:
            if hasattr(args, param):
                print(f"{param:25}: {getattr(args, param)}")

def parse_smart_contracts(filename):
    """
    Parse smart contract file and extract code fragments with labels
    Enhanced version with better error handling and validation
    
    Args:
        filename: Path to smart contract file
        
    Yields:
        tuple: (fragment_code, vulnerability_label)
    """
    print(f'Parsing smart contracts from: {filename}')
    
    # Pattern to detect file identifiers
    file_pattern = re.compile(r'^\d+\s+\d+\.sol$')
    
    # Statistics
    stats = {
        'total_contracts': 0,
        'vulnerable': 0,
        'safe': 0,
        'skipped': 0
    }
    
    with open(filename, "r", encoding="utf8") as file:
        fragment = []
        fragment_label = 0
        line_count = 0
        
        for line in file:
            line_count += 1
            stripped = line.strip()
            
            if not stripped:
                continue
            
            # Skip file identifiers
            if file_pattern.match(stripped):
                continue
                
            # Fragment separator
            if "-" * 40 in line and fragment:
                # Validate fragment
                if CONFIG['MIN_FRAGMENT_LENGTH'] <= len(fragment) <= CONFIG['MAX_FRAGMENT_LENGTH']:
                    yield fragment, fragment_label
                    stats['total_contracts'] += 1
                    if fragment_label == 1:
                        stats['vulnerable'] += 1
                    else:
                        stats['safe'] += 1
                else:
                    stats['skipped'] += 1
                    if len(fragment) < CONFIG['MIN_FRAGMENT_LENGTH']:
                        print(f"Warning: Skipping fragment at line {line_count} (too short: {len(fragment)} lines)")
                    else:
                        print(f"Warning: Skipping fragment at line {line_count} (too long: {len(fragment)} lines)")
                fragment = []
                
            # Label line
            elif stripped in ['0', '1']:
                fragment_label = int(stripped)
            # Code line
            else:
                fragment.append(stripped)
    
    # Handle last fragment
    if fragment and CONFIG['MIN_FRAGMENT_LENGTH'] <= len(fragment) <= CONFIG['MAX_FRAGMENT_LENGTH']:
        yield fragment, fragment_label
        stats['total_contracts'] += 1
        if fragment_label == 1:
            stats['vulnerable'] += 1
        else:
            stats['safe'] += 1
    
    # Print parsing statistics
    print(f"\nParsing Statistics:")
    print(f"- Total contracts parsed: {stats['total_contracts']}")
    print(f"- Vulnerable contracts: {stats['vulnerable']} ({stats['vulnerable']/max(1, stats['total_contracts'])*100:.1f}%)")
    print(f"- Safe contracts: {stats['safe']} ({stats['safe']/max(1, stats['total_contracts'])*100:.1f}%)")
    print(f"- Skipped fragments: {stats['skipped']}")

def debug_parse_smart_contracts(filename, max_fragments=5):
    """
    Enhanced debug version for parsing verification
    
    Args:
        filename: Path to smart contract file
        max_fragments: Maximum number of fragments to display
    """
    print(f"\n{'='*60}")
    print("DEBUG: PARSING ANALYSIS")
    print(f"{'='*60}")
    
    fragments_analyzed = 0
    vulnerability_patterns = {
        'call.value': 0,
        'send': 0,
        'transfer': 0,
        'delegatecall': 0,
        'balance': 0
    }
    
    for i, (fragment, label) in enumerate(parse_smart_contracts(filename)):
        # Analyze patterns
        fragment_text = ' '.join(fragment).lower()
        for pattern in vulnerability_patterns:
            if pattern in fragment_text:
                vulnerability_patterns[pattern] += 1
        
        if fragments_analyzed < max_fragments:
            print(f"\n--- Fragment {i+1} (Label: {label}) ---")
            print(f"Length: {len(fragment)} lines")
            
            # Show first and last lines
            if len(fragment) > 10:
                print("First 5 lines:")
                for j, line in enumerate(fragment[:5]):
                    print(f"  {j+1:2d}: {line}")
                print(f"  ... ({len(fragment) - 10} lines omitted)")
                print("Last 5 lines:")
                for j, line in enumerate(fragment[-5:], len(fragment)-5):
                    print(f"  {j+1:2d}: {line}")
            else:
                for j, line in enumerate(fragment):
                    print(f"  {j+1:2d}: {line}")
            
            # Check for pollution
            polluted_lines = [line for line in fragment if line.endswith('.sol')]
            if polluted_lines:
                print(f"⚠️  WARNING: Polluted lines detected: {polluted_lines}")
            else:
                print("✅ Fragment clean")
                
            # Analyze vulnerability indicators
            vuln_indicators = []
            if 'call.value' in fragment_text:
                vuln_indicators.append('call.value pattern')
            if 'msg.sender.call' in fragment_text:
                vuln_indicators.append('external call')
            if re.search(r'balance.*=.*-', fragment_text):
                vuln_indicators.append('balance modification')
                
            if vuln_indicators:
                print(f"🔍 Vulnerability indicators: {', '.join(vuln_indicators)}")
                
            fragments_analyzed += 1
    
    print(f"\n{'='*60}")
    print("VULNERABILITY PATTERN ANALYSIS:")
    for pattern, count in vulnerability_patterns.items():
        print(f"- {pattern}: {count} occurrences")
    print(f"{'='*60}")

def create_dataset(filename, args, cache_enabled=True):
    """
    Create enhanced vectorized dataset from smart contract file
    
    Args:
        filename: Path to smart contract file
        args: Parsed arguments with vectorization parameters
        cache_enabled: Whether to use caching
        
    Returns:
        pd.DataFrame: Dataset with enhanced vectors and labels
    """
    print("\n" + "=" * 60)
    print("ENHANCED DATASET CREATION")
    print("=" * 60)
    
    # Check cache
    cache_file = Path(CONFIG['CACHE_DIR']) / f"{Path(filename).stem}_enhanced_vectors_v2.pkl"
    
    if cache_enabled and cache_file.exists():
        print(f"Loading cached dataset from {cache_file}")
        dataset = pd.read_pickle(cache_file)
        print(f"Loaded {len(dataset)} samples from cache")
        return dataset
    
    # Collect fragments
    fragments = []
    vectorizer = EnhancedFragmentVectorizer(
        args.vec_length,
        use_augmentation=args.use_data_augmentation
    )
    
    print("Collecting code fragments with enhanced analysis...")
    start_time = time.time()
    
    # Progress tracking
    fragment_count = 0
    for fragment, label in parse_smart_contracts(filename):
        fragment_count += 1
        if fragment_count % 100 == 0:
            print(f"Processing fragment {fragment_count:4d}...", end="\r")
        
        vectorizer.add_fragment(fragment)
        fragments.append({"fragment": fragment, "label": label})
    
    collection_time = time.time() - start_time
    print(f"\nCollected {fragment_count} fragments in {collection_time:.2f}s")
    print(f"Forward slices: {vectorizer.forward_slices}")
    print(f"Backward slices: {vectorizer.backward_slices}")
    
    # Train enhanced Word2Vec model
    print("\nTraining enhanced Word2Vec model...")
    start_time = time.time()
    vectorizer.train_model()
    training_time = time.time() - start_time
    print(f"Word2Vec training completed in {training_time:.2f}s")
    
    # Print vocabulary statistics
    if args.vocab_stats:
        stats = vectorizer.get_vocabulary_stats()
        print(f"\nVocabulary Statistics:")
        print(f"- Total unique tokens: {stats['total_tokens']}")
        print(f"- Final vocabulary size: {stats['final_vocab_size']}")
        print(f"- Average fragment length: {stats['avg_fragment_length']:.2f}")
        print(f"- Average complexity score: {stats['avg_complexity']:.2f}")
        print(f"- Augmentation ratio: {stats['augmented_ratio']:.2%}")
        print(f"- Most common tokens:")
        for token, count in stats['most_common_tokens'][:10]:
            print(f"    {token}: {count}")
    
    # Vectorize fragments
    print("\nVectorizing fragments with enhanced method...")
    start_time = time.time()
    
    dataset = []
    for i, fragment_data in enumerate(fragments):
        if (i + 1) % 100 == 0:
            print(f"Vectorizing {i+1:4d}/{len(fragments)}", end="\r")
        
        vector = vectorizer.vectorize(fragment_data["fragment"])
        dataset.append({"vector": vector, "label": fragment_data["label"]})
    
    vectorization_time = time.time() - start_time
    print(f"\nVectorization completed in {vectorization_time:.2f}s")
    
    # Create DataFrame
    df = pd.DataFrame(dataset)
    
    # Save to cache
    if cache_enabled:
        print(f"Saving dataset to cache: {cache_file}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        df.to_pickle(cache_file)
    
    # Print dataset statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Vulnerable samples: {sum(df['label'] == 1)} ({sum(df['label'] == 1)/len(df)*100:.1f}%)")
    print(f"Safe samples: {sum(df['label'] == 0)} ({sum(df['label'] == 0)/len(df)*100:.1f}%)")
    print(f"Vector shape: {df.iloc[0]['vector'].shape}")
    
    # Analyze vector quality
    sample_vector = df.iloc[0]['vector']
    non_zero_ratio = np.count_nonzero(sample_vector) / sample_vector.size
    print(f"Vector non-zero ratio: {non_zero_ratio:.2%}")
    print(f"Vector mean magnitude: {np.mean(np.abs(sample_vector)):.4f}")
    print(f"Vector std deviation: {np.std(sample_vector):.4f}")
    
    return df

def plot_training_curves(history, save_path="training_curves.png", show_in_colab=True):
    """Enhanced plot training curves with better visualization"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot accuracy
    ax = axes[0, 0]
    ax.plot(history.history['accuracy'], label='Training', linewidth=2, marker='o', markersize=4)
    if 'val_accuracy' in history.history:
        ax.plot(history.history['val_accuracy'], label='Validation', linewidth=2, marker='s', markersize=4)
    ax.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Plot loss
    ax = axes[0, 1]
    ax.plot(history.history['loss'], label='Training', linewidth=2, marker='o', markersize=4)
    if 'val_loss' in history.history:
        ax.plot(history.history['val_loss'], label='Validation', linewidth=2, marker='s', markersize=4)
    ax.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot precision if available
    if 'precision' in history.history:
        ax = axes[1, 0]
        ax.plot(history.history['precision'], label='Training', linewidth=2)
        if 'val_precision' in history.history:
            ax.plot(history.history['val_precision'], label='Validation', linewidth=2)
        ax.set_title('Model Precision', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    # Plot recall if available
    if 'recall' in history.history:
        ax = axes[1, 1]
        ax.plot(history.history['recall'], label='Training', linewidth=2)
        if 'val_recall' in history.history:
            ax.plot(history.history['val_recall'], label='Validation', linewidth=2)
        ax.set_title('Model Recall', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Recall', fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Wide + TabTransformer Training Progress', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to: {save_path}")
    
    # Display in Colab
    if show_in_colab and 'google.colab' in sys.modules:
        plt.show()
    else:
        try:
            display(Image(filename=save_path))
        except:
            print(f"Plot saved but cannot display. Check: {save_path}")
    
    plt.close()
    
    # Print training summary
    print_training_summary(history)

def print_training_summary(history):
    """Print comprehensive training summary"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    # Final metrics
    final_metrics = {}
    for metric in ['accuracy', 'loss', 'precision', 'recall', 'auc']:
        if metric in history.history:
            final_train = history.history[metric][-1]
            final_metrics[f'train_{metric}'] = final_train
            
            if f'val_{metric}' in history.history:
                final_val = history.history[f'val_{metric}'][-1]
                final_metrics[f'val_{metric}'] = final_val
                gap = abs(final_train - final_val)
                
                print(f"\n{metric.capitalize()}:")
                print(f"  Training:   {final_train:.4f}")
                print(f"  Validation: {final_val:.4f}")
                print(f"  Gap:        {gap:.4f}")
    
    # Best epoch analysis
    if 'val_loss' in history.history:
        best_epoch = np.argmin(history.history['val_loss'])
        print(f"\nBest Epoch: {best_epoch + 1}")
        print(f"  Val Loss: {history.history['val_loss'][best_epoch]:.4f}")
        print(f"  Val Accuracy: {history.history['val_accuracy'][best_epoch]:.4f}")
    
    # Overfitting analysis
    if 'val_accuracy' in history.history:
        acc_gap = final_metrics.get('train_accuracy', 0) - final_metrics.get('val_accuracy', 0)
        if acc_gap > 0.1:
            print(f"\n⚠️  Warning: Potential overfitting detected (accuracy gap: {acc_gap:.4f})")
        elif acc_gap > 0.05:
            print(f"\n⚡ Mild overfitting observed (accuracy gap: {acc_gap:.4f})")
        else:
            print(f"\n✅ Good generalization (accuracy gap: {acc_gap:.4f})")

def plot_metrics_comparison(results, save_path="metrics_comparison.png", show_in_colab=True):
    """Enhanced metrics comparison plot"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Main metrics bar plot
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        results['accuracy'],
        results['precision'],
        results['recall'],
        results['f1_score']
    ]
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax1.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.9)')
    ax1.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Good (>0.8)')
    ax1.legend(loc='lower right')
    
    # Error rates plot
    error_metrics = ['False Positive\nRate', 'False Negative\nRate']
    error_values = [results['fp_rate'], results['fn_rate']]
    error_colors = ['#e74c3c', '#f39c12']
    
    bars2 = ax2.bar(error_metrics, error_values, color=error_colors, edgecolor='black', linewidth=1.5)
    
    for bar, value in zip(bars2, error_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylim(0, max(error_values) * 1.2)
    ax2.set_ylabel('Rate', fontsize=12)
    ax2.set_title('Error Rates Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Wide + TabTransformer Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison saved to: {save_path}")
    
    if show_in_colab and 'google.colab' in sys.modules:
        plt.show()
    else:
        try:
            display(Image(filename=save_path))
        except:
            print(f"Plot saved but cannot display. Check: {save_path}")
    
    plt.close()

def plot_confusion_matrix_detailed(results, save_path="confusion_matrix_analysis.png", show_in_colab=True):
    """Create detailed confusion matrix visualization"""
    plt.style.use('seaborn-v0_8-white')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract confusion matrix values
    cm = results.get('confusion_matrix', {})
    tn = cm.get('tn', 0)
    fp = cm.get('fp', 0)
    fn = cm.get('fn', 0)
    tp = cm.get('tp', 0)
    
    # Confusion matrix heatmap
    matrix = np.array([[tn, fp], [fn, tp]])
    im = ax1.imshow(matrix, interpolation='nearest', cmap='Blues')
    ax1.figure.colorbar(im, ax=ax1)
    
    # Labels
    ax1.set(xticks=np.arange(2),
           yticks=np.arange(2),
           xticklabels=['Safe', 'Vulnerable'],
           yticklabels=['Safe', 'Vulnerable'],
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, f'{matrix[i, j]}\n({matrix[i, j]/(matrix.sum())*100:.1f}%)',
                           ha="center", va="center", color="white" if matrix[i, j] > matrix.max()/2 else "black",
                           fontsize=14, fontweight='bold')
    
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Performance metrics by class
    safe_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    safe_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    vuln_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    vuln_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    metrics_data = {
        'Safe': [safe_precision, safe_recall],
        'Vulnerable': [vuln_precision, vuln_recall]
    }
    
    x = np.arange(2)
    width = 0.35
    
    ax2.bar(x - width/2, [safe_precision, vuln_precision], width, label='Precision', color='#3498db')
    ax2.bar(x + width/2, [safe_recall, vuln_recall], width, label='Recall', color='#2ecc71')
    
    ax2.set_ylabel('Score')
    ax2.set_title('Per-Class Performance')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Safe', 'Vulnerable'])
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (precision, recall) in enumerate(zip([safe_precision, vuln_precision], [safe_recall, vuln_recall])):
        ax2.text(i - width/2, precision + 0.01, f'{precision:.3f}', ha='center', va='bottom')
        ax2.text(i + width/2, recall + 0.01, f'{recall:.3f}', ha='center', va='bottom')
    
    plt.suptitle('Confusion Matrix and Per-Class Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix analysis saved to: {save_path}")
    
    if show_in_colab and 'google.colab' in sys.modules:
        plt.show()
    else:
        try:
            display(Image(filename=save_path))
        except:
            print(f"Plot saved but cannot display. Check: {save_path}")
    
    plt.close()

def train_with_kfold(dataset, args, k=5):
    """
    Train model using K-fold cross validation
    
    Args:
        dataset: DataFrame with vectors and labels
        args: Training arguments
        k: Number of folds
        
    Returns:
        Dict with aggregated results
    """
    print(f"\n{'='*60}")
    print(f"K-FOLD CROSS VALIDATION (k={k})")
    print(f"{'='*60}")
    
    # Prepare data
    X = np.stack(dataset['vector'].values)
    y = dataset['label'].values
    
    # Initialize k-fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # Results storage
    fold_results = []
    all_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*50}")
        print(f"FOLD {fold}/{k}")
        print(f"{'='*50}")
        
        # Create fold dataset
        fold_data = pd.DataFrame({
            'vector': [X[i] for i in train_idx],
            'label': y[train_idx]
        })
        
        # Train model
        model = WideTabTransformer(fold_data, args)
        history = model.train()
        results = model.evaluate()
        
        # Store results
        fold_results.append(results)
        all_histories.append(history)
        
        # Print fold results
        print(f"\nFold {fold} Results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
    
    # Aggregate results
    aggregated_results = {}
    for metric in fold_results[0].keys():
        if isinstance(fold_results[0][metric], (int, float)):
            values = [r[metric] for r in fold_results]
            aggregated_results[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    # Print summary
    print(f"\n{'='*60}")
    print("K-FOLD CROSS VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    for metric, stats in aggregated_results.items():
        if metric != 'confusion_matrix':
            print(f"\n{metric}:")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    return aggregated_results, all_histories

def save_results(results, args, save_path="results/experiment_results.json"):
    """Save experiment results with metadata"""
    experiment_data = {
        'timestamp': datetime.now().isoformat(),
        'configuration': vars(args),
        'results': results,
        'system_info': {
            'python_version': sys.version,
            'tensorflow_version': tf.__version__,
            'numpy_version': np.__version__,
            'platform': sys.platform
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(experiment_data, f, indent=4, default=str)
    
    print(f"\nResults saved to: {save_path}")

def main():
    """Main execution function"""
    IN_COLAB = 'google.colab' in sys.modules
    
    if IN_COLAB:
        print("🔵 Running in Google Colab environment")
        # Ensure matplotlib works properly in Colab
        import matplotlib
        matplotlib.use('module://ipykernel.pylab.backend_inline')
    
    # Parse arguments
    args = parameter_parser()
    
    # Print header and parameters
    print_header()
    print_parameters(args)
    
    # 🆕 DEBUG DU PARSING
    print(f"\n{'='*60}")
    print("VERIFICATION DU PARSING (MODE DEBUG)")
    print(f"{'='*60}")
    
    # Test du parsing avec debug
    debug_parse_smart_contracts(args.filename, max_fragments=3)
    
    # Demander confirmation avant de continuer (en mode interactif seulement)
    if sys.stdin.isatty():  # Vérifie si on est en mode interactif
        user_input = input("\nLe parsing semble-t-il correct ? (y/n) [y]: ").lower().strip()
        if user_input and user_input != 'y':
            print("Parsing interrompu. Vérifiez les données d'entrée.")
            return
    else:
        print("\nMode non-interactif détecté, continuation automatique...")
    
    # Prepare dataset path
    base_name = os.path.splitext(os.path.basename(args.filename))[0]
    dataset_path = f"config/train_data/{base_name}_enhanced_vectors.pkl"
    
    # Create data directory
    os.makedirs("config/train_data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    print(f"\nDataset path: {dataset_path}")
    
    # Load or create dataset
    if os.path.exists(dataset_path):
        print("Loading existing enhanced dataset...")
        dataset = pd.read_pickle(dataset_path)
        print("Enhanced dataset loaded successfully!")
    else:
        print("Creating new enhanced dataset...")
        dataset = create_dataset(args.filename, args)
        print(f"Saving enhanced dataset to {dataset_path}...")
        dataset.to_pickle(dataset_path)
        print("Enhanced dataset saved successfully!")
    
    # Model training and evaluation
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize model
    print("Initializing Wide + TabTransformer model...")
    model = WideTabTransformer(dataset, args)
    model.get_model_summary()
    
    # Train model
    history = model.train()
    
    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time:.2f}s")
    
    # Plot training curves
    print("\n" + "=" * 60)
    print("GENERATING TRAINING CURVES")
    print("=" * 60)
    
    plot_training_curves(
        history, 
        save_path=f"plots/{base_name}_training_curves.png",
        show_in_colab=IN_COLAB
    )

    # Evaluate model
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    results = model.evaluate()

    # Plot metrics comparison
    print("\n" + "=" * 60)
    print("GENERATING METRICS COMPARISON")
    print("=" * 60)
    
    plot_metrics_comparison(
        results,
        save_path=f"plots/{base_name}_metrics_comparison.png",
        show_in_colab=IN_COLAB
    )
    
    # Plot confusion matrix analysis
    print("\n" + "=" * 60)
    print("GENERATING CONFUSION MATRIX ANALYSIS")
    print("=" * 60)
    
    plot_confusion_matrix_detailed(
    results,
    save_path=f"plots/{base_name}_confusion_matrix_analysis.png",
    show_in_colab=IN_COLAB)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Architecture: Wide + TabTransformer")
    print(f"Vectorization: Enhanced with vulnerability patterns")
    print(f"Dataset: {args.filename}")
    print(f"Vulnerability Type: {args.vt}")
    print(f"Training Parameters:")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Transformer Layers: {args.num_transformer_layers}")
    print(f"  - Attention Heads: {args.num_heads}")
    print(f"  - Embedding Dim: {args.embedding_dim}")
    print(f"  - Dropout: {args.dropout}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Final Accuracy: {results['accuracy']:.4f}")
    print(f"Final F1-Score: {results['f1_score']:.4f}")
    print(f"\nPlots saved in: plots/")
    
    if IN_COLAB:
        print("\n📊 All plots have been displayed inline in Colab")
    
    print("=" * 60)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 60)

if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass  

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)