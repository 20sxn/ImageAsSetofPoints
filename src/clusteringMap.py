from contextClusters import *
from contextClusters_original_mask import *
from utils import *

PATH = "datasets"
SAVEPATH = "./"

def get_clustering_maps(model, idx, head_choice, alpha=0.1, img_size=32):
    up = torch.nn.Upsample(size=(img_size,img_size))

    X,y = test_dataset.__getitem__(idx)
    X = X.reshape(-1,3,img_size,img_size)
    img,_ = img_dataset.__getitem__(idx)
    img = img.reshape(-1,3,img_size,img_size)
    X = X.to(device)
    X.device
    
    _, masks = model.getmask(X)
    
    clusters=[]
    for c,m in zip(head_choice,masks):
        clusters.append(m[c])
        
    colors = []
    for c in clusters:
        colors.append(torch.argmax(c,dim=1))
    title = "Image source"
    plt.title(title)
    plt.imshow(rearrange(img[0],'c h w -> h w c').cpu().numpy(),alpha= 1)
    plt.show()
    for i,c in enumerate(colors):
        t = up((c[None]).float())
        title = "Clustering Map du Stage n°"+str(i)
        plt.title(title)
        plt.imshow(rearrange(img[0],'c h w -> h w c').cpu().numpy(),alpha= 1)
        plt.imshow(t.reshape(img_size,img_size).cpu().numpy(),alpha = alpha)
        
        plt.show()


if __name__ == "__main__":
    cifar = True
    device = "cuda"

    if cifar:
        # CoC-supertiny
        n_blocks = [4,6,4]
        embedding_sizes = [5,32,128]
        head_counts = [4,4,8]
        head_dims = [16,16,16]
        mlp_ratios = [16,8,4]
        folds = [2,1,1]
        num_classes = 10
        model = Model(embedding_sizes,num_classes,n_blocks=n_blocks,head_counts=head_counts,head_dims=head_dims,mlp_ratios=mlp_ratios,dropout=0.5,droppath=0.2)

        model.load_state_dict(torch.load("models/model_super_small.pth"))
        model.eval()
        model = model.to(device)

        test_dataset = datasets.CIFAR10(PATH, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor (), # Transform the PIL image to a torch.Tensor
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]))

        img_dataset = datasets.CIFAR10(PATH, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor (), # Transform the PIL image to a torch.Tensor
            # No normalization for plots
        ]))

        head_choice = [torch.randint(h,(1,)) for h in head_counts]
        idx = torch.randint(len(test_dataset),(1,))
        print(idx)
        get_clustering_maps(model,idx,0.3,img_size=32)

    else:
        # CoC-tiny
        head_counts = [4,4,8,8]
        pretrained_model = coc_tiny()
        embedding_sizes = [32, 64, 196, 320]
        num_classes = 101
        model = CaltechModel(pretrained_model, embedding_sizes, num_classes)

        model.load_state_dict(torch.load("models/caltech-tiny.cpkt"))
        model.eval()
        model = model.to(device)

        transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
        dataset = datasets.Caltech101(PATH,download=True,transform=transform)

        head_choice = [torch.randint(h,(1,)) for h in head_counts]
        idx = torch.randint(len(dataset),(1,))
        print(idx)
        get_clustering_maps(model,idx,0.3,img_size=224)

