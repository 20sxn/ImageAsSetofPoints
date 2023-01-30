from contextClusters import *

def main_10(model,n_epochs,device):
    cuda = True

    batch_size = 128
    train_dataset = datasets.CIFAR10(PATH, train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor (), # Transform the PIL image to a torch.Tensor
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(),
        transforms.RandomHorizontalFlip(),
    ]))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=True, pin_memory=cuda, num_workers=1)
    
    test_dataset = datasets.CIFAR10(PATH, train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor (), # Transform the PIL image to a torch.Tensor
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]))
    test_loader = torch.utils.data.DataLoader(test_dataset,
                        batch_size=batch_size, shuffle=True, pin_memory=cuda, num_workers=1)
    

    Loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    optim = torch.optim.AdamW(model.parameters())
    
    List_Loss = []
    List_acc = []
    
    for epoch in range(n_epochs):
        model.train()
        losses = []
        for X,y in train_loader:
            X,y = X.to(device), y.to(device)
            optim.zero_grad()
            
            yhat = model(X)

            l = Loss(yhat,y)
            l.backward()
            
            optim.step()
            losses.append(l.detach().cpu())
        if epoch % 10 == 0:
            acc = 0
            total = 0
            model.eval()
            for X,y in test_loader:
                X,y = X.to(device), y.to(device)
                yhat = model(X)

                y_pred = torch.argmax(yhat,dim = 1)
                acc += torch.sum(y==y_pred)
                total += len(y)
                #l = Loss(yhat,y)
            List_acc.append(acc.item()/total)
            
        List_Loss.append(torch.mean(torch.stack(losses)).detach().cpu())
        if epoch % 10 == 0:
            print(f"epoch n°{epoch} : loss = {List_Loss[-1]}, acc = {acc.item()/total}")
        else:
            print(f"epoch n°{epoch} : loss = {List_Loss[-1]}")
    
    plt.figure()
    plt.plot(range(n_epochs),List_Loss)
    plt.show()
    plt.close()

    
    plt.figure()
    plt.plot(range(int(np.ceil(n_epochs/10))),List_acc)
    plt.show()
    plt.close()