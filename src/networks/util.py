import time
import torch


def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f"{item:>6}")
    print(f"______\n{sum(params):>6}")


def run_batch_convolution_neural_network(
    train_loader,
    test_loader,
    model,
    criterion,
    optimizer,
    batch_size: int,
    epochs: int = 20,
    is_debug: bool = False,
) -> tuple:
    start_time = time.time()
    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []
    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0

        # Run the training batches
        for b, (X_train, y_train) in enumerate(train_loader):
            b += 1

            # Apply the model
            y_pred = model(X_train)  # we don't flatten X-train here
            loss = criterion(y_pred, y_train)

            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print interim results
            if b % 5 == 0 and is_debug:
                print(
                    f"epoch: {i:2}  batch: {b:4} [{batch_size*b:6}/1009]  loss: {loss.item():10.8f}  \
    accuracy: {trn_corr.item()*100/(batch_size*b):7.3f}%"
                )

        train_losses.append(loss.item())
        train_correct.append(trn_corr)

        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):

                # Apply the model
                y_val = model(X_test)

                # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1]
                tst_corr += (predicted == y_test).sum()

        loss = criterion(y_val, y_test)
        test_losses.append(loss)
        test_correct.append(tst_corr)

    # print the time elapsed
    elapsed_time = time.time() - start_time
    if is_debug:
        print(f"\nDuration: {elapsed_time:.0f} seconds")
    return (elapsed_time, train_losses, test_losses, train_correct, test_correct)
