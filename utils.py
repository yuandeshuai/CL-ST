import scipy.io as io
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.calibration import CalibratedClassifierCV
import cleanlab
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
from tqdm import tqdm

def CL_ST(X, Y, base_model,num):
    model=base_model
    label_quality_scores_model_spa_rf=np.zeros([num,X.shape[0]])
    label_quality_scores_model_spa_rf_cali=np.zeros([num,X.shape[0]])
    for i in tqdm(range(num), desc="CL-ST Progress"):
        probabilities, probabilities_cali = get_sample_quality_multi_division(X, Y, model, method='sigmoid')
        label_quality_scores_model = cleanlab.rank.get_label_quality_scores(
            labels=np.array(Y),
            pred_probs=probabilities,
            method="self_confidence"
        )
        label_quality_scores_model_spa_rf[i, :] = label_quality_scores_model
        label_quality_scores_model_cali = cleanlab.rank.get_label_quality_scores(
            labels=np.array(Y),
            pred_probs=probabilities_cali,
            method="self_confidence"
        )
        label_quality_scores_model_spa_rf_cali[i, :] = label_quality_scores_model_cali
    io.savemat('quality_label_random_spa_rf.mat',{'probabilities':probabilities,
                                                    'probabilities_cali':probabilities_cali,
                                                    'label_quality_scores_model_spa_rf':label_quality_scores_model_spa_rf,
                                                   'label_quality_scores_model_spa_rf_cali':label_quality_scores_model_spa_rf_cali})

    # mat_data = io.loadmat('quality_label_random_spa_rf.mat')
    # probabilities = mat_data['probabilities']
    # probabilities_cali = mat_data['probabilities_cali']
    # label_quality_scores_model_spa_rf = mat_data['label_quality_scores_model_spa_rf']
    # label_quality_scores_model_spa_rf_cali = mat_data['label_quality_scores_model_spa_rf_cali']

    # statistical testing, out-of-sample predicted probability estimated through random sampling follow a normal distribution
    label_quality=np.mean(label_quality_scores_model_spa_rf_cali, axis=0)
    #Calculation of Calibration Curve
    prob_true_clf, prob_pred_clf = calibration_curve(Y, probabilities[:, 1], n_bins=30)
    prob_true_calibrated, prob_pred_calibrated = calibration_curve(Y, probabilities_cali[:, 1], n_bins=30)
    # Calculate the Brilleis score
    brier_score_clf = brier_score_loss(Y, probabilities[:, 1])
    brier_score_calibrated = brier_score_loss(Y, probabilities_cali[:, 1])
    print(f"Brier score (uncalibrated): {brier_score_clf:.4f}")
    print(f"Brier score (calibrated): {brier_score_calibrated:.4f}")
    # Plot calibration curve
    plt.figure(figsize=(6, 4.5))
    plt.plot(prob_pred_clf, prob_true_clf, marker='o', linewidth=1, label='Uncalibrated')
    plt.plot(prob_pred_calibrated, prob_true_calibrated, marker='o', linewidth=1, label='Calibrated')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Ideally Calibrated')
    plt.xlabel('Mean Predicted Probability', fontsize=15, fontname='times new roman')
    plt.ylabel('Fraction of Positives', fontsize=15, fontname='times new roman')
    plt.title('Calibration Curves')
    plt.legend()
    plt.show()
    return label_quality




def get_sample_quality_multi_division(X, Y, base_model,method):
    # m = 0
    kf = KFold(n_splits=10, shuffle=True)
    probabilities2 = np.zeros((X.shape[0], len(np.unique(Y))))  # Initialize probability matrix
    probabilities2_cali = np.zeros((X.shape[0], len(np.unique(Y))))  # Initialize probability matrix
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        # m=m+1
        # print(m)
        X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=1 / 9)

        # Train the base model
        # base_model = ELMClassifier(n_hidden_units=400, alpha=0.00)
        base_model.fit(X_train, Y_train)

        # base_model probabilities
        probas = base_model.predict_proba(X_test)
        probabilities2[test_index] = probas

        # Calibrate the model
        calibrated_model = CalibratedClassifierCV(base_model, method=method, cv='prefit')
        calibrated_model.fit(X_validation, Y_validation)

        # Predict probabilities
        probas_cali = calibrated_model.predict_proba(X_test)
        probabilities2_cali[test_index] = probas_cali
    return probabilities2,probabilities2_cali

def aquire_index(label):
    kernerl_index_cell = []
    mask=label
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if mask[i, j] in (1, 2):
                seeds = [Point(i, j)]
                result = regionGrow(label, seeds, thresh=0.2, p=1)  #### result 包含当前颗粒范围的1
                # plt.imshow(result)
                # plt.show()
                nonzero_indices = np.transpose(np.nonzero(result))
                kernerl_index_cell.append(nonzero_indices)
                label_temp = label * result  #### result 包含当前颗粒范围的标签（1霉变或者2健康）
                mask = mask - label_temp
    return kernerl_index_cell

def aquire_pixel(train_img_selected,train_img_label_selected):
    train_img_selected_t = np.reshape(train_img_selected, [-1, train_img_selected.shape[-1]])
    nonzero_indices = np.nonzero(train_img_selected_t[:, 0])[0]
    pos = np.zeros([nonzero_indices.shape[0], 2])
    train_img_selected_pixel = np.zeros([nonzero_indices.shape[0], train_img_selected.shape[2]])
    train_img_label_selected_pixel = np.zeros([nonzero_indices.shape[0]])
    m = 0
    for i in range(train_img_label_selected.shape[0]):
        for j in range(train_img_label_selected.shape[1]):
            if train_img_label_selected[i, j] != 0:
                pos[m, 0], pos[m, 1] = i, j
                train_img_selected_pixel[m, :] = train_img_selected[i, j, :]
                train_img_label_selected_pixel[m] = train_img_label_selected[i, j]
                m = m + 1
    return train_img_selected_pixel,train_img_label_selected_pixel,pos.astype(np.int32)

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
    return connects

def regionGrow(img, seeds, thresh, p=1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    # seedMark = np.full((img.shape), np.nan)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while (len(seedList) > 0):
        currentPoint = seedList.pop(0)

        seedMark[currentPoint.x, currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))
    return seedMark


def comparing(label,train):
    mask = label
    correct_classmap_k = np.zeros_like(label)
    correct_classmap_kk = np.zeros_like(label)
    correct_classmap_kkk = np.zeros_like(label)
    k = 0.05;kk=0.1;kkk=0.15
    A_k = np.zeros((2, 2));A_kk = np.zeros((2, 2));A_kkk = np.zeros((2, 2))
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if mask[i, j] in (1, 2):
                seeds = [Point(i, j)]
                result = regionGrow(label, seeds, thresh=0.2, p=1)
                train_temp = train * result
                label_temp = label * result
                rate = np.count_nonzero(train_temp == 1) / np.count_nonzero(train_temp)
                if rate > k:
                    train_temp_map_k = np.where(train_temp != 0, 1, 0)
                else:
                    train_temp_map_k = np.where(train_temp != 0, 2, 0)
                correct_classmap_k = correct_classmap_k + train_temp_map_k
                if np.amax(label_temp) == 1:
                    if rate > k:
                        A_k[1, 1] = A_k[1, 1] + 1
                    else:
                        A_k[1, 0] = A_k[1, 0] + 1
                if np.amax(label_temp) == 2:
                    if rate > k:
                        A_k[0, 1] = A_k[0, 1] + 1
                    else:
                        A_k[0, 0] = A_k[0, 0] + 1


                if rate > kk:
                    train_temp_map_kk = np.where(train_temp != 0, 1, 0)
                else:
                    train_temp_map_kk = np.where(train_temp != 0, 2, 0)
                correct_classmap_kk = correct_classmap_kk + train_temp_map_kk
                if np.amax(label_temp) == 1:
                    if rate > kk:
                        A_kk[1, 1] = A_kk[1, 1] + 1
                    else:
                        A_kk[1, 0] = A_kk[1, 0] + 1
                if np.amax(label_temp) == 2:
                    if rate > kk:
                        A_kk[0, 1] = A_kk[0, 1] + 1
                    else:
                        A_kk[0, 0] = A_kk[0, 0] + 1


                if rate > kkk:
                    train_temp_map_kkk = np.where(train_temp != 0, 1, 0)
                else:
                    train_temp_map_kkk = np.where(train_temp != 0, 2, 0)
                correct_classmap_kkk = correct_classmap_kkk + train_temp_map_kkk
                if np.amax(label_temp) == 1:
                    if rate > kkk:
                        A_kkk[1, 1] = A_kkk[1, 1] + 1
                    else:
                        A_kkk[1, 0] = A_kkk[1, 0] + 1
                if np.amax(label_temp) == 2:
                    if rate > kkk:
                        A_kkk[0, 1] = A_kkk[0, 1] + 1
                    else:
                        A_kkk[0, 0] = A_kkk[0, 0] + 1

                mask = mask - label_temp
        print(i)
    return A_k,A_kk,A_kkk


def comparing_quick(train_label_kernel, train_pred_kernel,train_label_kernel_index,PMP):
    A_k = np.zeros((2, 2));
    k=PMP
    for i in range(len(train_label_kernel_index)):
        pixel_temp_index = train_label_kernel_index[i]
        pixel_pre=np.zeros([np.shape(pixel_temp_index)[0],1])
        pixel_label = np.zeros([np.shape(pixel_temp_index)[0],1])
        for j in range(np.shape(pixel_temp_index)[0]):
            pixel_pre[j]=train_pred_kernel[pixel_temp_index[j,0],pixel_temp_index[j,1]]
            pixel_label[j]=train_label_kernel[pixel_temp_index[j,0],pixel_temp_index[j,1]]

        rate = np.count_nonzero(pixel_pre == 1) / np.count_nonzero(pixel_label)
        ## 计算阈值 k
        if np.amax(pixel_label) == 1:
            if rate > k:
                A_k[1, 1] = A_k[1, 1] + 1
            else:
                A_k[1, 0] = A_k[1, 0] + 1
        if np.amax(pixel_label) == 2:
            if rate > k:
                A_k[0, 1] = A_k[0, 1] + 1
            else:
                A_k[0, 0] = A_k[0, 0] + 1
    return A_k


def comparing_quick_all_thred(train_label_kernel, train_pred_kernel,train_label_kernel_index):
    A_k = np.zeros((100, 4))
    for i in range(len(train_label_kernel_index)):
        pixel_temp_index = train_label_kernel_index[i]
        pixel_pre=np.zeros([np.shape(pixel_temp_index)[0],1])
        pixel_label = np.zeros([np.shape(pixel_temp_index)[0],1])
        for j in range(np.shape(pixel_temp_index)[0]):
            pixel_pre[j]=train_pred_kernel[pixel_temp_index[j,0],pixel_temp_index[j,1]]
            pixel_label[j]=train_label_kernel[pixel_temp_index[j,0],pixel_temp_index[j,1]]

        rate = np.count_nonzero(pixel_pre == 1) / np.count_nonzero(pixel_label)
        for mm in range(100):
            if np.amax(pixel_label) == 1:
                if rate > mm*0.01:
                    A_k[mm, 3] = A_k[mm, 3] + 1
                else:
                    A_k[mm, 2] = A_k[mm, 2] + 1
            if np.amax(pixel_label) == 2:
                if rate > mm*0.01:
                    A_k[mm, 1] = A_k[mm, 1] + 1
                else:
                    A_k[mm, 0] = A_k[mm, 0] + 1
    return A_k

def comparing_quick_map(train_label_kernel, train_pred_kernel,train_label_kernel_index,rat):
    kernel_map=np.zeros_like(train_label_kernel)
    for i in range(len(train_label_kernel_index)):
        pixel_temp_index = train_label_kernel_index[i]
        pixel_pre=np.zeros([np.shape(pixel_temp_index)[0],1])
        pixel_label = np.zeros([np.shape(pixel_temp_index)[0],1])
        for j in range(np.shape(pixel_temp_index)[0]):
            pixel_pre[j]=train_pred_kernel[pixel_temp_index[j,0],pixel_temp_index[j,1]]
            pixel_label[j]=train_label_kernel[pixel_temp_index[j,0],pixel_temp_index[j,1]]

        rate = np.count_nonzero(pixel_pre==1) / np.count_nonzero(pixel_label)

        if rate > rat:
            for k in range(np.shape(pixel_temp_index)[0]):
                kernel_map[pixel_temp_index[k,0],pixel_temp_index[k,1]]=1
        else:
            for k in range(np.shape(pixel_temp_index)[0]):
                kernel_map[pixel_temp_index[k, 0], pixel_temp_index[k, 1]] = 2

    return kernel_map

