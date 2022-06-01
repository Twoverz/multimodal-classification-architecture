from numpy import interp
from torchvision import transforms as T
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import torch
import os
import itertools
import cv2
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import *
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
from torchvision import transforms
from HSACon_merge import FeedForward, FeedForward_vgg

######################################3###
class MyDataset(Dataset):
    def __init__(self, train_path, transform=None):
        self.vf = os.listdir(train_path + '/train_vf')
        self.fund = os.listdir(train_path + '/train_fund')
        assert len(self.vf) == len(self.fund), 'Number does not match'
        self.transform = transform
        self.images_and_labels = []  # 存储图像和标签路径
        for i in range(len(self.vf)):  # 图片加载进去了
            self.images_and_labels.append(
                (train_path + '/train_vf/' + self.vf[i], train_path + '/train_fund/' + self.fund[i]))

    def __getitem__(self, item):
        vf_path, fund_path = self.images_and_labels[item]
        vf = cv2.imread(vf_path)
        vf = cv2.resize(vf, (224, 224))
        fund = cv2.imread(fund_path)
        fund = cv2.resize(fund, (224, 224))
        img = np.concatenate((vf,fund),axis=2)#224 224 5
        # img = np.transpose(img, (2, 0, 1))

        #img = Image.fromarray(img)

        if 'a' in vf_path.split('/')[-1]:
            label = 0
        elif 'b' in vf_path.split('/')[-1]:
            label = 1
        elif 'c' in vf_path.split('/')[-1]:
            label = 2
        elif 'd' in vf_path.split('/')[-1]:
            label = 3
        if self.transform is not None:
            img = self.transform(img)
        #print(vf_path.split('/')[-1], label)
        #img = img / 255
        return img, label

    def __len__(self):
        return len(self.vf)


# train
transform2 = transforms.Compose([transforms.Resize(224), transforms.RandomResizedCrop(224), transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225])])
transform = transforms.Compose([transforms.ToTensor()])
# test

# ---------------------------数据集--------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = MyDataset('qgy_fund/vf_train/vf_fund', transform)
train_dataloader = DataLoader(train_data, 16, shuffle=True, num_workers=0)


# ------------------定义网络---------------------------------
# 载入预训练的型
model = models.squeezenet1_1(pretrained=True)
model.features[0] = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=(3, 3), stride=(2, 2))
#model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=4, kernel_size=(1, 1), stride=(1, 1))
model.classifier[1] = FeedForward(dim=676,hidden_dim=1024,p=13)

#model = models.vgg19_bn(pretrained=True)
#model.features[0] = nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#model.classifier = FeedForward_vgg(25088,4096,4)
#model.classifier[6] = nn.Linear(in_features=4096, out_features=4, bias=True)



model.to(device)

# ------------------优化方法，损失函数--------------------------------------------------
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.001)
loss_fc = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
'''
# ------------------训练--------------------------------------------------------------
num_epoch = 70
loss = []
iteration = []
acc = []
for epoch in range(1, num_epoch + 1):
    losslist = []
    acclist = []
    train_correct = 0
    train_total = 0
    a = 0
    b = 0
    ave_loss = 0

    for i, (data, label) in enumerate(train_dataloader):
        #print(data.shape,label.shape)
        inputs = Variable(data)
        labels = Variable(label)
        inputs = inputs.to(device)
        labels = labels.to(device)
        model.train()
        # forward
        outputs = model(inputs)
        # outputs = torch.sigmoid(outputs)

        #print(labels.shape,outputs.shape)
        # loss
        loss_train = loss_fc(outputs, labels)
        losslist.append(loss_train.item())
        # forward update
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # 统计
        # train_loss += loss.item()
        # print(outputs.type,labels.type)
        train_correct += (torch.max(outputs, 1)[1] == labels).sum().item()
        train_total += labels.size(0)
        acclist.append((train_correct / train_total))
        # print(train_correct / train_total)

        print('[{},{}] train_loss = {:.5f} train_acc = {:.5f} '.format(
            epoch + 1, i + 1, loss_train.item(), train_correct / train_total))

    # scheduler.step()
    if epoch % 5 == 0:  # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        torch.save(model, 'train_pth/class_qgym5_sq_epoch_{}.pth'.format(epoch))
        print('train_pth/class_qgym5_sq_epoch_{}.pth saved!'.format(epoch))

    for x in range(len(losslist)):
        a += losslist[x]
    ave_loss = a / (len(losslist))
    print(ave_loss)

    for y in range(len(acclist)):
        b += acclist[y]
    ave_acc = b / (len(acclist))
    print(ave_acc)

    loss.append(ave_loss)
    acc.append((ave_acc))
    iteration.append(epoch)

plt.figure()
plt.title("Loss Curve")
plt.plot(iteration, loss)
plt.ylabel("Train loss")
plt.xlabel("Iters")
plt.draw()
plt.show()

plt.figure()
plt.title("Accuracy Curve")
plt.plot(iteration, acc)
plt.ylabel("Train accuracy")
plt.xlabel("Iters")
plt.draw()
plt.show()

print('Train finish!')

'''
################################ceshi
transform1 = transforms.Compose([transforms.Resize(224), T.CenterCrop(224), transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225])])

class TestDataset(Dataset):
    def __init__(self, train_path, transform=None):
        self.vf = os.listdir(train_path + '/test_vf')
        self.fund = os.listdir(train_path + '/test_fund')
        assert len(self.vf) == len(self.fund), 'Number does not match'
        self.transform = transform
        self.images_and_labels = []  # 存储图像和标签路径
        for i in range(len(self.vf)):  # 图片加载进去了
            self.images_and_labels.append(
                (train_path + '/test_vf/' + self.vf[i], train_path + '/test_fund/' + self.fund[i]))

    def __getitem__(self, item):
        vf_path, fund_path = self.images_and_labels[item]
        vf = cv2.imread(vf_path)
        vf = cv2.resize(vf, (224, 224))
        fund = cv2.imread(fund_path)
        fund = cv2.resize(fund, (224, 224))
        img = np.concatenate((vf,fund),axis=2)#224 224 5
        #img = np.transpose(img, (2, 0, 1))

        #img = Image.fromarray(img)

        if 'a' in vf_path.split('/')[-1]:
            label = 0
        elif 'b' in vf_path.split('/')[-1]:
            label = 1
        elif 'c' in vf_path.split('/')[-1]:
            label = 2
        elif 'd' in vf_path.split('/')[-1]:
            label = 3
        if self.transform is not None:
            img = self.transform(img)
        #print(vf_path.split('/')[-1], label)
        #img = img / 255
        return img, label

    def __len__(self):
        return len(self.vf)

test_data = TestDataset('qgy_fund/vf_train/vf_fund', transform)
checkpoint_path = 'train_pth/class_qgym5_sq_epoch_25.pth'

model.eval()

test_dataloader = DataLoader(test_data, 1, shuffle=False, num_workers=0)

model = torch.load(checkpoint_path)
predictions = []
label1s = []
out = []
for i, (data, label) in enumerate(test_dataloader):
    data = Variable(data)
    label = Variable(label)
    data = data.to(device)
    label = label.to(device)
    #with torch.no_grad():
        #input = Variable(data)
    probability = model(data)  # torch.Size([1, 2])
    probability = torch.sigmoid(probability)  # [:, 0]#.data.tolist()
    #probability = torch.nn.functional.softmax(probability, dim=1)
    doc = probability.cpu().detach().numpy()#.flatten()
    pro = probability.cpu().detach().numpy().flatten()  # float32
    pro = np.argmax(pro, axis=0)
    pro = pro.flatten()

    lab = label.cpu().detach().numpy().flatten()  # float32

    out.append(doc)
    predictions.append(pro)
    label1s.append(lab)

pred = np.concatenate(predictions, 0)  # numpy.ndarray
plab = np.concatenate(label1s, 0)
pdoc = np.concatenate(out, 0)


def plot_confusion_matrix(cmtx, num_classes, class_names=None, figsize=(6.4, 4.8)):
    """
    A function to create a colored and labeled confusion matrix matplotlib figure
    given true labels and preds.
    Args:
        cmtx (ndarray): confusion matrix.
        num_classes (int): total number of classes.
        class_names (Optional[list of strs]): a list of class names.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].

    Returns:
        img (figure): matplotlib figure.
    """
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure(figsize=figsize)
    plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] > threshold else "black"
        plt.text(
            j,
            i,
            format(cmtx[i, j]) if cmtx[i, j] != 0 else "0",
            horizontalalignment="center",
            color=color,
        )

    # plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig('pltpicture/' + "qgy_matrix.png")
    return figure


confusion = confusion_matrix(plab, pred)#4class
plot_confusion_matrix(confusion, num_classes=4)
print(confusion)


##################################评价指标
acc = accuracy_score(plab, pred)
f1 = f1_score(plab, pred,average='weighted')
kappa = cohen_kappa_score(plab, pred)
jaccard = jaccard_score(plab, pred,average='weighted')
recall = recall_score(plab, pred,average='weighted')
print('acc =',acc)
print('F1 score =',f1)
print('kappa score =',kappa)
print('jaccard_score =',jaccard)
print('recall score:',recall)


########四分类roc
y_pred_probabilities=pdoc
classnames=[]
for classname in plab:
    classnames.append(classname)


y_actual_binary = label_binarize(plab, classes=[0, 1, 2, 3])
y_pred_binary = y_pred_probabilities#label_binarize(y_pred_probabilities, classes=[0, 1, 2, 3, 4])
n_classes=4
lw=2


#计算每一类的ROC曲线和ROC面积

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_actual_binary[:, i], y_pred_binary[:, i])#[:, i]
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_actual_binary.ravel(), y_pred_binary.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#计算宏观平均ROC曲线和ROC面积

#第一个汇总所有假阳性率
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#然后在此点对所有ROC曲线进行插值
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

#最后求平均值并计算AUC

mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


#绘制所有ROC曲线

#plt.figure(figsize=(5, 5))  ########四个合在一起的roc曲线


colors = itertools.cycle(['red', 'blue', 'green', 'yellow'])
for i, color in zip(range(n_classes), colors):
    plt.figure(figsize=(5, 5))
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='Area Under the Curve (AUC = %0.4f)' % roc_auc[i])

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of class{}'.format(i))
    plt.legend(loc="best")
    plt.savefig('pltpicture/' + "qgy_matrix{}.png".format(i))
    #plt.show()

print('-----------------------------------------------------------------------------------------')


muticlass = multilabel_confusion_matrix(plab, pred)
print(muticlass)


for i in range(len(muticlass)):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    TP += muticlass[i][0, 0]
    TN += muticlass[i][1, 1]
    FN += muticlass[i][0, 1]
    FP += muticlass[i][1, 0]
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    spe = TN / (TN + FP)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('*****************The {} class********************'.format(i))
    print("{}th Precision: ".format(i) + str(p))
    print("{}th Sensitivity:".format(i) + str(r))
    print("{}th Specificity:".format(i) + str(spe))
    print("{}th F1 score (F-measure): ".format(i) + str(F1))
    print("{}th Accuracy: ".format(i) + str(acc))

