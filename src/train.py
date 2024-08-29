# Define the Vision Transformer model
model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10)  # Adjust `num_classes` to match your dataset

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Criterion, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training Loop with Validation
for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar for training
    train_progress = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{10}", unit="batch")
    
    for images, labels in train_progress:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        train_progress.set_postfix({
            "Train Loss": running_loss / len(train_loader),
            "Accuracy": 100 * correct / total
        })

    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Log metrics to Weights & Biases
    wandb.log({
        "Epoch": epoch+1,
        "Train Loss": avg_train_loss,
        "Train Accuracy": 100 * correct / total,
        "Validation Loss": avg_val_loss,
        "Validation Accuracy": val_accuracy
    })
