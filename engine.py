"""
Training and evaluation loops shared by the fusion model.
"""

import numpy as np
import torch


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, side, labels in loader:
        images, side, labels = images.to(device), side.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, side)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = outputs.argmax(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    y_true, y_pred, y_probs = [], [], []

    for images, side, labels in loader:
        images, side, labels = images.to(device), side.to(device), labels.to(device)

        outputs = model(images, side)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        probs = torch.softmax(outputs, dim=1)
        predicted = probs.argmax(1)

        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, np.array(y_true), np.array(y_pred), np.array(y_probs)


def fit(model, train_loader, test_loader, optimizer, criterion, device,
        epochs, patience, checkpoint_path):
    """Full training loop with early stopping on test accuracy."""
    best_test_accuracy = 0.0
    counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        test_loss, test_acc, *_ = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print("Best model updated.")
        else:
            counter += 1
            print(f"Early stopping counter: {counter}/{patience}")

        if counter >= patience:
            print("Early stopping triggered.")
            break

    print("\nTraining Completed.")
    print(f"Best Test Accuracy: {best_test_accuracy:.2f}%")
    return best_test_accuracy
