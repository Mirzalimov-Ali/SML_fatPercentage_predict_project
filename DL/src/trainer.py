import torch

class Trainer:
    def __init__(self, model, criterion, optimizer, patience=15):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.train_losses = []
        self.val_losses = []

        self.best_loss = float("inf")
        self.patience = patience
        self.counter = 0

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):

            # TRAIN
            self.model.train()
            train_loss = 0

            for xb, yb in train_loader:

                self.optimizer.zero_grad()

                logits = self.model(xb)
                loss = self.criterion(logits, yb)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * xb.size(0)

            train_loss /= len(train_loader.dataset)
            self.train_losses.append(train_loss)


            # VALIDATION
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for xb, yb in val_loader:

                    logits = self.model(xb)
                    loss = self.criterion(logits, yb)

                    val_loss += loss.item() * xb.size(0)

            val_loss /= len(val_loader.dataset)
            self.val_losses.append(val_loss)


            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )


            # EARLY STOPPING
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.counter = 0

            else:
                self.counter += 1

            if self.counter >= self.patience:
                print("Early stopping triggered")
                break

        return self.train_losses, self.val_losses