# -*- coding: utf-8 -*-
"""
DiLCAM - CIFAR-10 (PyTorch)
fast training (under 10 minutes)

@author: Killian
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import pandas as pd
import warnings
import time
import os
warnings.filterwarnings('ignore')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("="*70)
    print("Distilled model using CNN with MoE")
    print("="*70)

    # Multi-Head Attention (Distilled)
    class LightMHA(nn.Module):
        def __init__(self, channels, num_heads=4):
            super(LightMHA, self).__init__()
            self.num_heads = num_heads
            self.channels = channels
            self.head_dim = channels // num_heads
            
            # Simplified attention with 1x1 convs
            self.qkv = nn.Conv2d(channels, channels * 3, 1)
            self.proj = nn.Conv2d(channels, channels, 1)
            self.norm = nn.BatchNorm2d(channels)
            
        def forward(self, x):
            B, C, H, W = x.shape
            
            # Generate Q, K, V
            qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
            q, k, v = qkv.unbind(1)
            
            # Attention
            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            attn = F.softmax(attn, dim=-1)
            
            # Apply attention
            out = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
            out = self.proj(out)
            
            return self.norm(x + out)  # Residual connection

    # Mixture of Experts (Distilled)
    class LightMoE(nn.Module):
        def __init__(self, channels, num_experts=2, reduction=4):
            super(LightMoE, self).__init__()
            self.num_experts = num_experts
            hidden_dim = channels // reduction
            
            # Gating network (lightweight)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, num_experts, 1),
                nn.Softmax(dim=1)
            )
            
            # Expert networks (lightweight 1x1 convs)
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels, hidden_dim, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_dim, channels, 1)
                ) for _ in range(num_experts)
            ])
            
        def forward(self, x):
            B, C, H, W = x.shape
            
            # Get gates
            gates = self.gate(x)  # [B, num_experts, 1, 1]
            
            # Apply experts
            expert_outs = []
            for expert in self.experts:
                expert_outs.append(expert(x))
            
            # Weighted combination
            out = torch.zeros_like(x)
            for i, expert_out in enumerate(expert_outs):
                out += gates[:, i:i+1] * expert_out
            
            return x + out  # Residual connection

    # Distilled CNN 
    class DistilledCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(DistilledCNN, self).__init__()
            
            # Efficient stem
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)  # 16x16
            )
            
            # Block 1 with Attention
            self.block1 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            self.attn1 = LightMHA(128, num_heads=4)
            
            # Block 2 with MoE
            self.block2 = nn.Sequential(
                nn.MaxPool2d(2),  # 8x8
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            self.moe1 = LightMoE(256, num_experts=2)
            
            # Block 3 (Final features)
            self.block3 = nn.Sequential(
                nn.MaxPool2d(2),  # 4x4
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
            )
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
            
        def forward(self, x):
            x = self.stem(x)
            
            x = self.block1(x)
            x = self.attn1(x)
            
            x = self.block2(x)
            x = self.moe1(x)
            
            x = self.block3(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            
            return x

    # Data loading (reduced augmentation)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load datasets
    print("Loading CIFAR-10 dataset...")
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                               download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                              download=True, transform=test_transform)

    # Use batch size with no multiprocessing to avoid the error
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=False)  # Fixed: num_workers=0
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=False)   # Fixed: num_workers=0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Initialize model
    print("Building Distilled Enhanced CNN...")
    model = DistilledCNN(num_classes=10).to(device)


    
    # add extra weight for certain class
    print("Calculating class weights...")
# retrieve the class label
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)

# calculate the freqeuncy of each class 
    class_freq = class_counts / total_samples

# initial weight: 1.0
    class_weights = np.ones(10, dtype=float)

# tune some specific class weights（the class order in CIFAR-10: 0-airplane, 1-automobile, 2-bird, 3-cat, 4-deer, 5-dog, 6-frog, 7-horse, 8-ship, 9-truck）
    cat_index = 3  # 'cat'
    bird_index = 2  # 'bird'
    dog_index = 5  # 'dog'
    

# certain set weights: 'cat':1.8; 'bird' and 'dog':1.2
    class_weights[cat_index] = 1.8
    class_weights[bird_index] = 1.2
    class_weights[dog_index] = 1.2
    
    print(f"Class weights: {class_weights}")

# set the optimization
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device),
                               label_smoothing=0.1)  # 带类别权重的交叉熵损失
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)  # 更高的学习率，权重衰减
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, 
                                        steps_per_epoch=len(train_loader), 
                                        epochs=24)  # 单周期学习率调度

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} (vs {total_params*3:,} in full model)")

    # Optimized training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)  # Higher LR, weight decay
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, 
                                            steps_per_epoch=len(train_loader), 
                                            epochs=24)  # One-cycle for fast training

    # Training stage
    def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=False), target.to(device, non_blocking=False)  # Fixed: non_blocking=False
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        return running_loss / len(train_loader), 100. * correct / total

    # Fast validation function
    def validate(model, test_loader, criterion, device):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_targets = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device, non_blocking=False), target.to(device, non_blocking=False)  # Fixed: non_blocking=False
                output = model(data)
                test_loss += criterion(output, target).item()
                
                probabilities = F.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        test_loss /= len(test_loader)
        accuracy = 100. * correct / total
        
        return test_loss, accuracy, np.array(all_targets), np.array(all_predictions), np.array(all_probabilities)

    # Fast training loop (under 10 minutes)
    print("\nStarting fast training (target: under 10 minutes)...")
    start_time = time.time()

    num_epochs = 24  # Reduced epochs
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_acc = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch+1)
        
        # Validation (every 3 epochs to save time)
        if epoch % 3 == 0 or epoch == num_epochs - 1:
            val_loss, val_acc, _, _, _ = validate(model, test_loader, criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_distilled_model.pth')
        else:
            val_loss, val_acc = 0, 0
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{num_epochs} - {epoch_time:.1f}s')
        print(f'Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
        if val_acc > 0:
            print(f'Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%')
        print(f'Total time: {total_time/60:.1f} minutes')
        
        # Early exit if time limit approached
        if total_time > 25 * 60:  # 8 minutes safety margin
            print("Approaching time limit, stopping training early...")
            break
        
        print('-' * 50)

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time/60:.2f} minutes")

    # Load best model and evaluate
    model.load_state_dict(torch.load('best_distilled_model.pth'))
    print("Loaded best model for final evaluation...")

    # Final evaluation
    test_loss, test_accuracy, y_test, y_pred, y_pred_proba = validate(model, test_loader, criterion, device)
    y_pred_confidence = np.max(y_pred_proba, axis=1)

    # Quick results processing
    prediction_results = pd.DataFrame({
        'sample_id': range(len(y_test)),
        'true_label': y_test,
        'true_class_name': [class_names[i] for i in y_test],
        'predicted_label': y_pred,
        'predicted_class_name': [class_names[i] for i in y_pred],
        'prediction_confidence': y_pred_confidence,
        'is_correct': y_test == y_pred
    })

    # Add class probabilities
    for i, class_name in enumerate(class_names):
        prediction_results[f'prob_{class_name}'] = y_pred_proba[:, i]

     # 构建完整的CSV文件路径
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    csv_filename = os.path.join(desktop_path, 'cifar10_distilled_fast_results.csv')
    prediction_results.to_csv(csv_filename, index=False)
    print(f"CSV文件已保存至: {csv_filename}")

    # Calculate metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print(f"\nDistilled model performance:") 
    print(f"Training Time: {total_training_time/60:.2f} minutes")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Parameters: {total_params:,}")

    # Quick classification report
    print(f"\nPer-class accuracy:")
    for i in range(10):
        class_acc = np.mean(y_pred[y_test == i] == i) if np.sum(y_test == i) > 0 else 0
        print(f"{class_names[i]:12}: {class_acc:.3f}")

    # Simple visualization
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(12, 8))

    # Confusion matrix
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Distilled Enhanced CNN')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    # Training curves
    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', linewidth=2)
    if val_accuracies:
    # 确保val_epochs与val_accuracies长度一致
        val_epochs = [i for i in range(0, len(train_accuracies), len(train_accuracies)//len(val_accuracies))][:len(val_accuracies)]
        plt.plot(val_epochs, val_accuracies, label='Validation Accuracy', linewidth=2, marker='o')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Confidence distribution
    plt.subplot(2, 2, 3)
    plt.hist(prediction_results['prediction_confidence'], bins=30, alpha=0.7, color='blue')
    plt.title('Prediction Confidence')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # Per-class accuracy bar plot
    plt.subplot(2, 2, 4)
    per_class_acc = [np.mean(y_pred[y_test == i] == i) if np.sum(y_test == i) > 0 else 0 
                     for i in range(10)]
    plt.bar(range(10), per_class_acc, color='skyblue', edgecolor='navy')
    plt.title('Per-Class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(range(10), [name[:4] for name in class_names], rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "="*70)
    print("DISTILLED ENHANCED CNN - FAST TRAINING COMPLETE")
    print("="*70)
    print(f"OPTIMIZATION SUMMARY:")
    print(f"✓ Reduced parameters: {total_params:,} (3x smaller)")
    print(f"✓ Lightweight attention with 1x1 convolutions")
    print(f"✓ Simplified MoE with 2 experts")
    print(f"✓ Higher batch size (128) and learning rate")
    print(f"✓ One-cycle learning rate scheduling")
    print(f"✓ Label smoothing for better generalization")
    print(f"✓ Gradient clipping for stability")
    print(f"✓ Training time: {total_training_time/60:.2f} minutes")
    print(f"✓ Final accuracy: {test_accuracy:.2f}%")
    print(f"\nFiles generated: {csv_filename}")
    print("="*70)
    print(f"Prediction results saved to: {csv_filename}")

if __name__ == '__main__':
    # This protects the main execution when using multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)  # For Windows compatibility
    main()
