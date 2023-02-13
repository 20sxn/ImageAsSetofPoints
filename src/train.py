from contextClusters import *
from contextClusters_original import *
from utils import *

PATH = "datasets"
SAVEPATH = "./"

def main_cifar(model, n_epochs, device, batch_size=64):
    train_loader, test_loader = load_cifar10(PATH,batch_size)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optim = torch.optim.AdamW(model.parameters())
    
    List_Train_Loss = []
    List_Train_acc = []
    List_Test_acc = []
    max_acc = 0.0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        losses = []
        train_acc, train_total = 0.0, 0
        for X,y in train_loader:
            X,y = X.to(device), y.to(device)
            optim.zero_grad()
            
            yhat = model(X)

            l = criterion(yhat,y)
            l.backward()
            
            optim.step()

            ypred = torch.argmax(yhat,dim = 1)
            train_acc += torch.sum(y==ypred).detach().cpu()
            train_total += len(y)
            losses.append(l.detach().cpu())
        List_Train_acc.append(train_acc.item()/train_total)

        # Testing every 10 epochs
        if epoch % 10 == 0:
            test_acc, test_total = 0.0, 0
            model.eval()
            for X,y in test_loader:
                X,y = X.to(device), y.to(device)
                yhat = model(X)

                ypred = torch.argmax(yhat,dim = 1)
                test_acc += torch.sum(y==ypred)
                test_total += len(y)
            List_Test_acc.append(test_acc.item()/test_total)

            # Saving the best model
            if test_acc.item() > max_acc:
                max_acc = test_acc.item()
                torch.save(model.state_dict(), SAVEPATH+'models/cifar_best-tiny.cpkt')
            
        List_Train_Loss.append(torch.mean(torch.stack(losses)).detach().cpu())

        # Display
        if epoch % 10 == 0:
            print(f"epoch n°{epoch} : loss = {List_Train_Loss[-1]}, acc = {test_acc.item()/test_total}")
        else:
            print(f"epoch n°{epoch} : loss = {List_Train_Loss[-1]}")
    
    # Saving log
    torch.save(List_Train_acc, SAVEPATH+"log/Train_acc_cifar.pt")
    torch.save(List_Test_acc, SAVEPATH+"log/Test_acc_cifar.pt")
    torch.save(List_Train_Loss, SAVEPATH+"log/Train_loss_cifar.pt")

    # Show plots
    plt.figure()
    plt.plot(range(n_epochs),List_Train_Loss)
    plt.title("Evolution of the training loss")
    plt.savefig(SAVEPATH+"log/Train_loss_cifar.svg")
    plt.show()
    plt.close()
    
    plt.figure()
    plt.plot(range(int(np.ceil(n_epochs/10))),List_Test_acc)
    plt.title("Evolution of the test performance")
    plt.savefig(SAVEPATH+"log/Test_acc_cifar.svg")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(range(int(np.ceil(n_epochs))),List_Train_acc)
    plt.title("Evolution of the train performance")
    plt.savefig(SAVEPATH+"log/Train_acc_cifar.svg")
    plt.show()
    plt.close()
    
def main_caltech(model, n_epochs, device, batch_size=64, img_size=128):
    train_loader, test_loader = load_caltech101(PATH, batch_size, img_size)    

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optim = torch.optim.AdamW(model.parameters())
    
    List_Train_Loss = []
    List_Train_acc = []
    List_Test_acc = []
    max_acc = 0.0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        losses = []
        train_acc, train_total = 0.0, 0
        for X,y in train_loader:
            X,y = X.to(device), y.to(device)
            optim.zero_grad()
            
            yhat = model(X)
            l = criterion(yhat,y)
            l.backward()
            
            optim.step()

            ypred = torch.argmax(yhat,dim = 1)
            train_acc += torch.sum(y==ypred).detach().cpu()
            train_total += len(y)
            losses.append(l.detach().cpu())
        List_Train_acc.append(train_acc.item()/train_total)

        # Testing every 10 epochs
        if epoch % 10 == 0:
            test_acc, test_total = 0.0, 0
            model.eval()
            for X,y in test_loader:
                X,y = X.to(device), y.to(device)
                yhat = model(X)

                ypred = torch.argmax(yhat,dim = 1)
                test_acc += torch.sum(y==ypred)
                test_total += len(y)
            List_Test_acc.append(test_acc.item()/test_total)

            # Saving the best model
            if test_acc.item() > max_acc:
                max_acc = test_acc.item()
                torch.save(model.state_dict(), SAVEPATH+'models/caltech_best-tiny.cpkt')
            
        List_Train_Loss.append(torch.mean(torch.stack(losses)).detach().cpu())

        # Display
        if epoch % 10 == 0:
            print(f"epoch n°{epoch} : loss = {List_Train_Loss[-1]}, acc = {test_acc.item()/test_total}")
        else:
            print(f"epoch n°{epoch} : loss = {List_Train_Loss[-1]}")
    
    # Saving log
    torch.save(List_Train_acc, SAVEPATH+"log/Train_acc_caltech.pt")
    torch.save(List_Test_acc, SAVEPATH+"log/Test_acc_caltech.pt")
    torch.save(List_Train_Loss, SAVEPATH+"log/Train_loss_caltech.pt")

    # Showing plots
    plt.figure()
    plt.plot(range(n_epochs),List_Train_Loss)
    plt.title("Evolution of the training loss")
    plt.savefig(SAVEPATH+"log/Train_loss_caltech.svg")
    plt.show()
    plt.close()
    
    plt.figure()
    plt.plot(range(int(np.ceil(n_epochs/10))),List_Test_acc)
    plt.title("Evolution of the test performance")
    plt.savefig(SAVEPATH+"log/Test_acc_caltech.svg")
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(range(int(np.ceil(n_epochs))),List_Train_acc)
    plt.title("Evolution of the train performance")
    plt.savefig(SAVEPATH+"log/Train_acc_caltech.svg")
    plt.show()
    plt.close()
    


if __name__ == "__main__":
    device = 'cuda'
    batch_size = 64

    cifar = True
    if cifar:
        # CoC-supertiny
        n_blocks = [4,6,4]
        embedding_sizes = [5,32,128]
        head_counts = [4,4,8]
        head_dims = [16,16,16]
        mlp_ratios = [16,8,4]
        folds = [2,1,1]
        num_classes = 10
        model = Model(embedding_sizes,num_classes,n_blocks=n_blocks,head_counts=head_counts,head_dims=head_dims,mlp_ratios=mlp_ratios,dropout=0.5,droppath=0.2).to(device)
        # print(summary(model2,(3,32,32)))

        start = time.time()
        main_cifar(model, 301, device, batch_size)
        stop = time.time()
        print(stop-start)

    else:
        # CoC-tiny
        pretrained_model = coc_tiny()
        pretrained_model.load_state_dict(torch.load(SAVEPATH+'models/tiny_model_best.pth.tar')["state_dict"])
        embedding_sizes = [32, 64, 196, 320]
        num_classes = 101
        model = CaltechModel(pretrained_model, embedding_sizes, num_classes).to(device)
        # print(summary(model,(3,224,224)))

        # CoC-small
        # pretrained_model = coc_small()
        # pretrained_model.load_state_dict(torch.load(SAVEPATH+'models/small_model_best.pth.tar')["state_dict"])
        # embedding_sizes = [64, 128, 320, 512]
        # num_classes = 101
        # model = CaltechModel(pretrained_model, embedding_sizes, num_classes)

        start = time.time()
        main_caltech(model, 21, device, batch_size, img_size=224)
        stop = time.time()
        print(stop-start)

