import torch

def train(model, train_loader, criterion, optimizer, epoch, device, writer, global_step, tgt_vocab_size):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        src = batch['input_ids'].to(device)
        tgt = batch['labels'].to(device)
        
        # Shift target for teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:].contiguous().view(-1)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(src, tgt_input)
        outputs = outputs.contiguous().view(-1, tgt_vocab_size)
        
        # Compute loss
        loss = criterion(outputs, tgt_output)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log the loss every 100 batches
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / 100
            print(f'Epoch [{epoch}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {avg_loss:.4f}')
            writer.add_scalar('Loss/Train', avg_loss, global_step)
            total_loss = 0  # Reset total loss for the next logging interval
        
        global_step += 1
    return global_step


def validate(model, val_loader, criterion, device, writer, epoch, tgt_vocab_size):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            src = batch['input_ids'].to(device)
            tgt = batch['labels'].to(device)
            
            # Shift target for teacher forcing
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:].contiguous().view(-1)
            
            # Forward pass
            outputs = model(src, tgt_input)
            outputs = outputs.view(-1, tgt_vocab_size)
            
            # Compute loss
            loss = criterion(outputs, tgt_output)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    print(f'Validation Loss after Epoch [{epoch}]: {avg_loss:.4f}')
    writer.add_scalar('Loss/Validation', avg_loss, epoch)
    return avg_loss