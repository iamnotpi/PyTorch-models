### Simple train function for PyTorch neural nets (assume that the data has already been shuffled)

import torch

def train(model, X, y, batch_size=32, epochs=1, val_split=0.0, loss_fn=None, optimizer=None):
    # For visualization
    train_loss_array = []
    val_loss_array = []
  
    # Split the training set and the validation set
    val_size = int(val_split * len(X))
    X_train = X[val_size:]
    y_train = y[val_size:]

    batches = len(X_train) // batch_size # Total number of batches
    loss = 0.0

    for epoch in range(epochs):
        model.train() # Put the model to training mode
        for batch in range(batches):
            X_train_batch = X_train[batch * batch_size: (batch + 1) * batch_size]
            y_train_batch = y_train[batch * batch_size: (batch + 1) * batch_size]
            y_logits = model(X_train_batch).squeeze()
            batch_loss = loss_fn(y_logits, y_train_batch) 
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss
        loss /= batches

        if val_split != 0:
            X_val = X[:val_size]
            y_val = y[:val_size]
            model.eval() # Evaluation mode
            with torch.inference_mode():
                y_val_logits = model(X_val).squeeze()
                val_loss = loss_fn(y_val_logits, y_val)
                print(f"Epochs: {epoch + 1} | Training loss: {loss:.4f} | Validation loss: {val_loss:.4f}")

            train_loss_array.append(loss.item())
            val_loss_array.append(val_loss.item())
        
    return train_loss_array, val_loss_array if val_split != 0 else train_loss_array
