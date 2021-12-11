import os
import numpy as np
import pandas as pd
import argparse
import SimpleITK as sitk

from Models.generator import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

from matplotlib import pyplot as plt
#%%
#classification_report
target_names = ['MSA', 'PID', 'PSP']
#%%
def calculate_ROCAUC(Y_ture, Y_prob):
    ROCAUC = roc_auc_score(Y_ture, Y_prob) if len(np.unique(Y_ture)) > 1 else 0.0
    return ROCAUC
    
def plot_ROCAUC(y_test, y_score, n_classes):
    def plot_figure(fpr, tpr, lw, figureTitle):
        plt.figure()
        plt.plot(fpr[lw], tpr[lw], color='darkorange', lw=lw+1, label='ROC curve (area = %0.2f)' % roc_auc[lw])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw+1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic: {}'.format(figureTitle))
        plt.legend(loc="lower right")
        plt.show()
        return True

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plot_figure(fpr, tpr, lw=0, figureTitle=target_names[0])
    plot_figure(fpr, tpr, lw=1, figureTitle=target_names[1])
    plot_figure(fpr, tpr, lw=2, figureTitle=target_names[2])
    
def performace_evaluate(groundTruth,predictedResult):
    """
    calculate the TP, FP, TN, FN
    """    
    eplison = 1e-6
    perDict = dict()
    conf_matrix = confusion_matrix(groundTruth, predictedResult)
    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    TN = conf_matrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN+eplison)
    # Specificity or true negative rate
    TNR = TN/(TN+FP+eplison) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP+eplison)
    # Negative predictive value
    NPV = TN/(TN+FN+eplison)
    # Fall out or false positive rate
    FPR = FP/(FP+TN+eplison)
    # False negative rate
    FNR = FN/(TP+FN+eplison)
    # False discovery rate
    FDR = FP/(TP+FP+eplison)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN+eplison)
    
    perDict['FP'] = FP
    perDict['FN'] = FN
    perDict['TP'] = TP
    perDict['TN'] = TN
    perDict['TPR'] = TPR
    perDict['TNR'] = TNR
    perDict['PPV'] = PPV
    perDict['NPV'] = NPV
    perDict['FPR'] = FPR
    perDict['FNR'] = FNR
    perDict['FDR'] = FDR
    perDict['ACC'] = ACC
    
    print('Sensitivity/recall:'+ str(TPR))
    print('Specificity:'+ str(TNR))
    print('Precision/positive predictive value:'+ str(PPV))
    print('Negative predictive value:'+ str(NPV))
    
    return perDict

def evaluate(projectRoot):
    # ground truth
    Y_test = np.load(os.path.join(projectRoot, 'testLabels.npy'))
    Y_test = np.squeeze(Y_test)
    Y_test = np.int64(Y_test)
    nb_classes = len(np.unique(Y_test))
    y_test = to_categorical(Y_test, nb_classes)
    print (Y_test)
    print (Y_test.dtype)

    # predict result
    y_predict = np.load(os.path.join(projectRoot,'Y_predict.npy'))
    # print y_predict
    Y_predict = np.argmax(y_predict, axis=1)
    print (Y_predict)
    print (Y_predict.dtype)

    # ROCAUC
    for item in target_names:
        position = target_names.index(item)
        print('ROCAUC_{}:{}'.format(item, calculate_ROCAUC(Y_ture=y_test[:,position], Y_prob = y_predict[:,position])))
    # plot ROCAUC
    plot_ROCAUC(y_test, y_predict, n_classes=3)

    # classification report
    print(classification_report(Y_test, Y_predict, target_names=target_names))
    # Sensitivity, Specificity, PPV, NPV
    perDict = performace_evaluate(Y_test,Y_predict)
    with open(os.path.join(projectRoot,'evaResult.txt'),mode='w') as file_handle:
        for item in target_names:
            position = target_names.index(item)
            file_handle.write('ROCAUC_{}:{}'.format(item, calculate_ROCAUC(Y_ture=y_test[:,position], Y_prob = y_predict[:,position])))
        
        file_handle.write(classification_report(Y_test, Y_predict, target_names=target_names))
        
        file_handle.write('Sensitivity/recall:'+ str(perDict['TPR'])+'\n')
        file_handle.write('Specificity:'+ str(perDict['TNR'])+'\n')
        file_handle.write('Precision/positive predictive value:'+ str(perDict['PPV'])+'\n')
        file_handle.write('Negative predictive value:'+ str(perDict['NPV'])+'\n')

#%%    
def main():
    parser = argparse.ArgumentParser(description = "PDDNET command line tool")
    parser.add_argument("--project_folder", type=str, help = "project folder to save the output data.")
    parser.add_argument('--CV_fold', type=int, default=6, help='cross validation fold')
    parser.add_argument('--currentFold', type=int, default=1, help='current training fold')
    args = parser.parse_args()

    projectRoot = os.path.join(args.project_folder, 'data_{0}_{1}'.format(args.CV_fold, args.currentFold))
    evaluate(projectRoot)

if __name__ == '__main__':
    main()