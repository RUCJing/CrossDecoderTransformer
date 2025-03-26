import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from text_featuring import text_feature, CSVDataset
import matplotlib as mpl
from matplotlib import rcParams
from params import EVAL_FILE_PATH, MODEL_PATH, TEST_BATCH_SIZE
zhfont1 = mpl.font_manager.FontProperties(fname="./font/simhei.ttf") 

def accuracy(outputs, labels):
    preds = torch.max(outputs, dim=1)[1]
    return (preds == labels).sum().item() / len(labels)


def test_step(model, testloader):
    model.eval()
    val_loss, val_acc = 0, 0
    y_preds = np.array([])
    all_labels = np.array([])
    pred_list = []
    with torch.no_grad():
        for texts, X_cat, X_num, labels, speakers in tqdm(testloader):
            texts, X_cat, X_num, labels, speakers = texts.to(device), X_cat.to(device), X_num.to(device), labels.to(device), speakers.to(device)
            outputs = model(texts, X_cat, X_num, speakers)
            val_acc += accuracy(outputs, labels)
            y_pred = F.softmax(outputs, dim=1).detach().cpu().numpy()
            pred_list += np.argmax(y_pred, axis=1).tolist()
            if len(y_preds) == 0: y_preds = y_pred
            else: y_preds = np.vstack([y_preds, y_pred])
            labels = labels.detach().cpu().numpy()
            all_labels = np.hstack([all_labels, labels])
    pred_score = y_preds[:,1]
    val_acc /= len(testloader)
    return val_acc, all_labels, pred_score, pred_list


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = torch.load(MODEL_PATH)
    model = model.to(device)
    evals = CSVDataset(EVAL_FILE_PATH)
    eval_dl = DataLoader(evals, batch_size=TEST_BATCH_SIZE)
    test_acc, true_label, pred_score, pred_label = test_step(model, eval_dl)
    label_dict = {True: 1, False: 0}
    for idx in range(len(true_label)):
        true_label[idx] = label_dict[true_label[idx]]
    print(classification_report(true_label, pred_label, digits=4))
    label_names = list(label_dict.keys())
    C = confusion_matrix(true_label, pred_label, labels=[0, 1])

    plt.matshow(C, cmap=plt.cm.Reds)  

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    num_local = np.array(range(len(label_names)))
    plt.xticks(num_local, label_names, rotation=90, fontproperties=zhfont1)  
    plt.yticks(num_local, label_names, fontproperties=zhfont1) 
    plt.ylabel('True Label', fontproperties=zhfont1)
    plt.xlabel('Pred Label', fontproperties=zhfont1)
    plt.savefig("./image/confusion_matrix.png")
    plt.show()
    
    plt.figure()
    auc = roc_auc_score(true_label, pred_score)
    fpr, tpr, thersholds = roc_curve(true_label, pred_score, pos_label=1)
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.3f})'.format(auc), lw=2)

    plt.xlim([-0.05, 1.05]) 
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate') 
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig("./image/ROC_curve.png")
