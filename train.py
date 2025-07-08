import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
import argparse
from datetime import datetime

from TinyLN import TinyLN
from data_utils import create_datasets, save_datasets, load_datasets

class AdvancedTrainer:
    def __init__(self, model, train_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.config = config
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1
        )
        
        # Training history
        self.train_history = {'loss': [], 'accuracy': []}
        
        # Best model tracking
        self.best_acc = 0
        self.best_f1 = 0
        self.best_model_state = None
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1} [{batch_idx}/{len(self.train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model"""
        raise NotImplementedError("Validation is no longer supported. Use test set for evaluation.")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            print(f'\nEpoch {epoch+1}/{self.config["epochs"]}')
            print('-' * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Update history
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            
            # Print results
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            
            # For simplicity, we'll track the training accuracy as the "best" if no validation.
            # In a real scenario, you'd likely evaluate on the test set periodically.
            if train_acc > self.best_acc:
                self.best_acc = train_acc
                # F1 score is not directly available from train_epoch, so keeping previous best_f1
                self.best_model_state = self.model.state_dict().copy()
                print(f'★ New best model (based on training accuracy)! Train Acc: {train_acc:.4f}')
        
        training_time = time.time() - start_time
        print(f'\nTraining completed in {training_time:.2f} seconds!')
        print(f'Best training accuracy: {self.best_acc:.4f}')
        print(f'Best F1 score: {self.best_f1:.4f} (Not updated during training)')
        
        return self.best_acc, self.best_f1, training_time
    
    def save_model(self, save_path):
        """Save the best model"""
        torch.save({
            'model_state_dict': self.best_model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'config': self.config,
            'best_acc': self.best_acc,
            'best_f1': self.best_f1
        }, save_path)
        print(f"Best model saved to {save_path}")
    
    def plot_training_history(self, save_path):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Training loss
        ax1.plot(self.train_history['loss'], label='Train Loss', color='blue')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Training accuracy
        ax2.plot(self.train_history['accuracy'], label='Train Acc', color='blue')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training history plot saved to {save_path}")

class ComprehensiveEvaluator:
    def __init__(self, model, test_loader, device, class_names=None):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names or ['No Leak', 'Leak']
    
    def evaluate(self):
        """Comprehensive evaluation on test set"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        inference_times = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                output = self.model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        avg_inference_time = np.mean(inference_times)
        model_params = sum(p.numel() for p in self.model.parameters())
        
        # Print results
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*80)
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Test F1 Score: {f1:.4f}")
        print(f"Average Inference Time: {avg_inference_time*1000:.3f} ms")
        print(f"Model Parameters: {model_params:,}")
        print(f"Model Size (approx): {model_params * 4 / 1024:.2f} KB")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_targets, all_predictions, 
                                  target_names=self.class_names))
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'model_parameters': model_params,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
    
    def plot_confusion_matrix(self, targets, predictions, save_path):
        """Plot confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train TinyLN for Pipeline Leakage Detection')
    parser.add_argument('--dataset', type=str, default='csv', choices=['A', 'B', 'csv'],
                       help='Dataset type (A, B, or csv)')
    parser.add_argument('--data_dir', type=str, default='.',
                       help='Directory containing dataset files')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--segment_length', type=int, default=1024,
                       help='Signal segment length')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"tinyln_dataset_{args.dataset}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Configuration based on paper specifications
    config = {
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'num_classes': 2,
        'in_channels': 1,
        'stem_channels': 64,
        'expansion_factor': 4,
        'num_irb_blocks': 1,
        'num_dcb_blocks': 1,
        'dataset_type': args.dataset,
        'segment_length': args.segment_length
    }
    
    # Save configuration
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create or load datasets
    print("Creating datasets...")
    try:
        train_dataset, test_dataset, preprocessor = create_datasets(
            args.data_dir, 
            dataset_type=args.dataset,
            segment_length=args.segment_length,
            scaler_type='standard'
        )
        
        # Save processed datasets
        save_datasets(train_dataset, test_dataset, preprocessor, 
                     os.path.join(save_dir, 'datasets'))
        
    except Exception as e:
        print(f"Error creating datasets: {e}")
        print("Please ensure your data directory contains the proper files.")
        return
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                            shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Initialize model
    model = TinyLN(
        in_channels=config['in_channels'],
        stem_channels=config['stem_channels'],
        num_classes=config['num_classes'],
        expansion_factor=config['expansion_factor'],
        num_irb_blocks=config['num_irb_blocks'],
        num_dcb_blocks=config['num_dcb_blocks']
    ).to(device)
    
    # Initialize trainer
    trainer = AdvancedTrainer(model, train_loader, device, config)
    
    # Train model
    print("Starting training...")
    best_train_acc, _, training_time = trainer.train() # f1 is not meaningful from training acc
    
    # Save model and plots
    model_path = os.path.join(save_dir, 'best_model.pth')
    trainer.save_model(model_path)
    
    plot_path = os.path.join(save_dir, 'training_history.png')
    trainer.plot_training_history(plot_path)
    
    # Load best model for evaluation
    model.load_state_dict(trainer.best_model_state)
    
    # Comprehensive evaluation
    evaluator = ComprehensiveEvaluator(model, test_loader, device)
    results = evaluator.evaluate()
    
    # Plot confusion matrix
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    evaluator.plot_confusion_matrix(results['targets'], results['predictions'], cm_path)
    
    # Save final results
    final_results = {
        'training_results': {
            'best_train_accuracy': best_train_acc,
            'training_time_seconds': training_time
        },
        'test_results': {
            'accuracy': results['accuracy'],
            'f1_score': results['f1_score'],
            'avg_inference_time_ms': results['avg_inference_time_ms'],
            'model_parameters': results['model_parameters']
        },
        'config': config
    }
    
    with open(os.path.join(save_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📁 Results saved to: {save_dir}")
    print(f"📊 Final test accuracy: {results['accuracy']:.4f}")
    print(f"📊 Final test F1 score: {results['f1_score']:.4f}")

if __name__ == "__main__":
    main()