def trainScatter(model, train_loader, optimizer, replace, scattering):
    model = model.to(device)
    model.train()
    num_examples = 0
    correct = 0
    train_loss = 0
    for data, target in tqdm(train_loader):
        target = replace(target)
        data, target = data.to(device), target.to(device)
        output = model(scattering(data))
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += F.cross_entropy(output, target, reduction='sum').item()
        num_examples += len(data)

    train_loss /= num_examples
    train_acc = 100. * correct / num_examples
    print(f'Train set: Average loss: {train_loss:.4f}, '
          f'Accuracy: {correct}/{num_examples} ({train_acc:.2f}%)')
    return train_loss, train_acc

def testScatter(model, test_loader, replace, scattering):
    device = next(model.parameters()).device
    
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            target = replace(target)
            data, target = data.to(device), target.to(device)
            output = model(scattering(data))
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_examples += len(data)

    test_loss /= num_examples
    test_acc = 100. * correct / num_examples

    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{num_examples} ({test_acc:.2f}%)')

    return test_loss, test_acc

def test_with_unlabel(model, test_loader, replace, scattering, gmm, mth_list, multi=False, prop=None):
    cnt_matrix = np.zeros((NUM_CLASS,NUM_CLASS))
    device = next(model.parameters()).device
    model.eval()
    num_examples = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            re_target = replace(target)
            data, re_target, target = data.to(device), re_target.to(device), target.to(device)
            output = model(scattering(data))
            preds = output.max(1, keepdim=True)[1]
            for idx,pred in enumerate(preds):
                if pred == NUM_CLASS-NUM_UNTARGET_CLASS:
                    feature = model.feature((scattering(data[idx]))).cpu().numpy()
                    if multi:
                        pred = torch.tensor(mth_list[multiGMM_pred(gmm, prop, feature)[0]]).to(device)
                    else:
                        pred = torch.tensor(mth_list[gmm.predict(feature)[0]]).to(device)
                cnt_matrix[target[idx].item(), pred.item()] += 1
                num_examples += 1
    return cnt_matrix

def multiGMM_pred(gmms, props, X):
    pred = np.zeros((X.shape[0], NUM_UNTARGET_CLASS), dtype=float)
    for i in range(NUM_UNTARGET_CLASS):
        pred[:,i] = props[i] * gmms[i].score_samples(X)
    return np.argmax(pred, axis=1)

def stage1_train():   
    """
    Load original dataset and perform trainsforms for the purpose of data argumentation
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transforms = [transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        normalize]
    train_set = datasets.CIFAR10(root="./data", train=True,
                                        transform=transforms.Compose(train_transforms),
                                        download=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                        transform=transforms.Compose(
                                            [transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train_set, 
                                            batch_size=512,
                                            shuffle=True,
                                            num_workers=6, 
                                            drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_set, 
                                            batch_size=1024,
                                            shuffle=False,
                                            num_workers=6, 
                                            drop_last=False)
    
    """
    Remove the labels of records from all those unseen classes
    """
    get_imgs = get_img(REPLACE_LST)
    unlabel_training_imgs, ground_truth_training = get_imgs(train_set)
    unlabel_training_dataset = UnLabel_Dataset(unlabel_training_imgs,
                                            ground_truth_training,
                                            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    unlabel_training_dataloader = torch.utils.data.DataLoader(unlabel_training_dataset,
                                                            batch_size=512,
                                                            shuffle=False,
                                                            num_workers=6,
                                                            drop_last=False)
    unlabel_test_imgs, ground_truth_test = get_imgs(test_set)
    unlabel_test_dataset = UnLabel_Dataset(unlabel_test_imgs,
                                        ground_truth_test,
                                        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    unlabel_test_dataloader = torch.utils.data.DataLoader(unlabel_test_dataset,
                                                            batch_size=512,
                                                            shuffle=False,
                                                            num_workers=6,
                                                            drop_last=False)

    """
    ScatterCNN training process: the data is first pased to a scatter layer and the output is used to train our CNN
    """
    model = ScatterCNN(classes=NUM_CLASS-NUM_UNTARGET_CLASS+1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                    momentum=0.9,
                                    nesterov=False)

    scheduler = MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)

    replace = replace_label(REPLACE_LST, NUM_CLASS-NUM_UNTARGET_CLASS)
    if device.type=="cuda":
        Scattering_pr = Scattering2D(J=2, shape=(32, 32)).cuda()
    else:
        Scattering_pr = Scattering2D(J=2, shape=(32, 32))

    train_loss_lst=[]
    train_acc_lst=[]
    val_loss_lst=[]
    val_acc_lst=[]
    for epoch in range(0, EPOCHS):
        print(f"\nEpoch: {epoch}")
        train_loss, train_acc = trainScatter(model, train_loader, optimizer, replace, Scattering_pr)
        test_loss, test_acc = testScatter(model, test_loader, replace, Scattering_pr)
        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_acc)
        val_loss_lst.append(test_loss)
        val_acc_lst.append(test_acc)
        scheduler.step()
    
    """
    Prepare training and validation dataset for the stage 2 GMM training.
    """
    model.eval()
    unlabel_features_training = torch.empty((0,64))
    grand_truth_training = torch.empty((0))
    for data, target in tqdm(unlabel_training_dataloader):
        data, target = data.to(device), target
        feature = model.feature((Scattering_pr(data)))
        unlabel_features_training = torch.cat((unlabel_features_training,feature.detach().cpu()), 0)
        grand_truth_training = torch.cat((grand_truth_training, target), 0)
    unlabel_features_training = unlabel_features_training.numpy()
    grand_truth_training = grand_truth_training.numpy()

    model.eval()
    unlabel_features_test = torch.empty((0,64))
    grand_truth_test = torch.empty((0))
    for data, target in tqdm(unlabel_test_dataloader):
        data, target = data.to(device), target
        feature = model.feature((Scattering_pr(data)))
        unlabel_features_test = torch.cat((unlabel_features_test,feature.detach().cpu()), 0)
        grand_truth_test = torch.cat((grand_truth_test, target), 0)
    unlabel_features_test = unlabel_features_test.numpy()
    grand_truth_test = grand_truth_test.numpy()
    
    return train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst, model, Scattering_pr, replace, unlabel_features_test, unlabel_features_training, grand_truth_training, grand_truth_test, test_set

def stage2_train(model, test_set, Scattering_pr, replace):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=6, drop_last=False)
    
    """
    All the prediction is using a single GMM
    """
    sin_GMM = GaussianMixture(n_components=NUM_UNTARGET_CLASS).fit(unlabel_features_training)
    pred = sin_GMM.predict(unlabel_features_test)

    match_map = np.zeros((len(REPLACE_LST),len(np.unique(pred))))
    for grand_truth_idx, tar in enumerate(REPLACE_LST):
        for pred_idx, p_tar in enumerate(np.unique(pred)):
            g_eq = grand_truth_test==tar
            pred_= pred==p_tar
            match_map[pred_idx, grand_truth_idx] = np.sum(g_eq&pred_)/1000
        
    match_lst = {}
    for ground_truth, i in enumerate(np.argmax(match_map,1)):
        match_lst[ground_truth]=REPLACE_LST[i]
    
    
    whole_match_map = test_with_unlabel(model,test_loader, replace, Scattering_pr, sin_GMM, match_lst)

    """
    Predict unseen labels with a single GMM and adpat N gmms for N different classes
    """
    rough_GMM = GaussianMixture(n_components=NUM_UNTARGET_CLASS).fit(unlabel_features_training)
    gmm_labels = rough_GMM.predict(unlabel_features_training)
    class_gmms = []
    prop = []
    for i in range(NUM_UNTARGET_CLASS):
        class_gmm = GaussianMixture(n_components=10).fit(unlabel_features_training[gmm_labels==i])
        class_gmms.append(class_gmm)
        prop.append(unlabel_features_training[gmm_labels==i].shape[0] / unlabel_features_training.shape[0])
    
    mgmm_pred = multiGMM_pred(class_gmms, prop, unlabel_features_test)
    mmatch_map = np.zeros((NUM_UNTARGET_CLASS, NUM_UNTARGET_CLASS))
    for grand_truth_idx, tar in enumerate(REPLACE_LST):
        for i in range(NUM_UNTARGET_CLASS):
            g_eq = grand_truth_test==tar
            pred_= mgmm_pred==i
            mmatch_map[i, grand_truth_idx] = np.sum(g_eq&pred_)/1000

    mmatch_lst = {}
    for ground_truth, i in enumerate(np.argmax(mmatch_map,1)):
        mmatch_lst[ground_truth]=REPLACE_LST[i]
    mwhole_match_map = test_with_unlabel(model,test_loader, replace, Scattering_pr, class_gmms, mmatch_lst, True, prop)



    return match_map, whole_match_map, mmatch_map, mwhole_match_map

def show_result(train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst, mm, wmm, mmm, mwmm):
    plt.figure(figsize=(20, 15))
    plt.subplot(321)
    plt.plot(train_loss_lst, label='Training loss')
    plt.plot(val_loss_lst, label='Validation loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(322)
    plt.plot(train_acc_lst, label='Training accuracy')
    plt.plot(val_acc_lst, label='validation accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(323)
    sns.heatmap(mm,vmin=0, vmax=1, annot=True, cmap="YlGnBu")
    plt.subplot(324)
    sns.heatmap(wmm,vmin=0, vmax=1, annot=True, cmap="YlGnBu")

    plt.subplot(325)
    sns.heatmap(mmm,vmin=0, vmax=1, annot=True, cmap="YlGnBu")
    plt.subplot(326)
    sns.heatmap(mwmm,vmin=0, vmax=1, annot=True, cmap="YlGnBu")

    plt.show()