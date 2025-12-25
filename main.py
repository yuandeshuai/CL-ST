import scipy.io as io
import numpy as np
import cleanlab
import matplotlib.pyplot as plt
from utils import comparing
from utils import comparing_quick
import cv2
import numpy as np
from utils import aquire_index
from utils import aquire_pixel
from sklearn.ensemble import RandomForestClassifier
from utils import get_sample_quality_multi_division
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from utils import CL_ST
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm
np.random.seed(7)

####################################################################################################### 1.读取数据
print("1.Loading data...")
# Loading raw data: including the training set, validation set, and test set, along with their corresponding weak labels
# The size of the hyperspectral data is H × W × B, and the size of the weak labels is H × W.
original_data = io.loadmat('original_data.mat')
train_img_selected= original_data['train_img_selected'] # training set
train_img_label_selected= original_data['train_img_label_selected'] # weak label of training set
validation_img_selected= original_data['validation_img_selected']
validation_img_label_selected= original_data['validation_img_label_selected']
test_img_selected= original_data['test_img_selected']
test_img_label_selected= original_data['test_img_label_selected']

# Use the regional growing algorithm to obtain the index of each peanut and its corresponding pixel coordinates.
# The returned result is in list format, with each cell including the coordinates of all pixels for a single peanut.
train_label_kernel_index=aquire_index(train_img_label_selected)
validation_label_kernel_index=aquire_index(validation_img_label_selected)
test_label_kernel_index=aquire_index(test_img_label_selected)

# Employing the regional growth algorithm to obtain pixel spectral data, weak labels, and their corresponding coordinate indices.
train_pixel,train_pixel_label,train_pixel_pos=aquire_pixel(train_img_selected,train_img_label_selected)
validation_pixel,validation_pixel_label,validation_pixel_pos=aquire_pixel(validation_img_selected,validation_img_label_selected)
test_pixel,test_pixel_label,test_pixel_pos=aquire_pixel(test_img_selected,test_img_label_selected)
print("1.Data loading complete")
####################################################################################################### 2.feature selection
print("2.Load feature bands...")
# Here we directly employ the band importance ranking obtained via SPA.
# You may also substitute other feature selection methods here.
spa_selected_all=[0,1,168,142,178,67,28,182,118,96,27,72,221,181,47,200,29,117,89,88,91,140,212,90,189,169,225,87,68,177,73,180,30,210,224,249,190,141,92,214,237,26,85,167,170,176,14,25,31,69,179,201,205,222,230,116,158,174,219,240,3,7,24,32,38,53,70,110,162,166,211,215,217,228,2,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,33,34,35,36,37,39,40,41,42,43,44,45,46,48,49,50,51,52,54,55,56,57,58,59,60,61,62,63,64,65,66,71,74,75,76,77,78,79,80,81,82,83,84,86,93,94,95,97,98,99,100,101,102,103,104,105,106,107,108,109,111,112,113,114,115,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,159,160,161,163,164,165,171,172,173,175,183,184,185,186,187,188,191,192,193,194,195,196,197,198,199,202,203,204,206,207,208,209,213,216,218,220,223,226,227,229,231,232,233,234,235,236,238,239,241,242,243,244,245,246,247,248,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287]
spa_selected = spa_selected_all[0:17] #

####################################################################################################### 3.Calculate Label Quality using CL-ST
print("3.Calculating Label Quality via CL-ST ...")
# Taking random forests as an example, X is the training set, and Y is the weakly labelled training set.
rf=RandomForestClassifier(n_estimators=100, max_depth=10,random_state=42)
X, Y = train_pixel[:, spa_selected], train_pixel_label.astype(np.int32)-1
num=20 # number of n in KT test,You can increase num to obtain a more statistically significant result.
label_quality=CL_ST(X, Y, rf,num)
# concatenate quality and label
quality_label = np.concatenate((np.reshape(train_pixel_label,[train_pixel.shape[0],1]),np.reshape(label_quality,[train_pixel.shape[0],1])), axis=1)
print("3. Label Quality Calculation completed")
####################################################################################################### 4. Visualisation of label quality
print("4.Visualization of Marking Quality ...")
# Data standardisation
scaler = StandardScaler()
normalized_data = scaler.fit_transform(train_pixel.T).T
normalized_data=normalized_data[:, spa_selected]
labels=train_pixel_label
color=label_quality # Colouring based on labels

# Principal Component Analysis
pca = PCA(n_components=15)
pca.fit(normalized_data)
components = pca.transform(normalized_data)

# Plot principal components and colour based on labels label
plt.figure()
n_samples=5000
mask0 = (labels == 1) & (label_quality >= 0.0)
mask1 = (labels == 2) & (label_quality >= 0.0)
plt.scatter(components[mask0, 0][:n_samples], components[mask0, 1][:n_samples],marker='^', s=5,  alpha=0.5, c=color[mask0][:n_samples], cmap='RdYlGn', label='STMP')
plt.scatter(components[mask1, 0][:n_samples], components[mask1, 1][:n_samples],marker='o', s=5,  alpha=0.5, c=color[mask1][:n_samples], cmap='RdYlGn',label='HP')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(loc='best', frameon=True, framealpha=0.7)
plt.grid(True)
plt.show()
# 查看高质量标记样本
plt.figure()
mask0 = (labels == 1) & (label_quality >= 0.99)
mask1 = (labels == 2) & (label_quality >= 0.99)
plt.scatter(components[mask0, 0], components[mask0, 1],marker='^', s=10,  alpha=0.5, c='r',label='STMP')
plt.scatter(components[mask1, 0], components[mask1, 1],marker='o', s=10,  alpha=0.5, c='b',label='HP')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.grid(True)
plt.legend(loc='best', frameon=True, framealpha=0.7)
plt.tight_layout()
plt.show()

####################################################################################################### 4.统计颗粒级结果
print("4.Flip low-quality labels and compute kernel-scale classification results. ...")
from utils import  comparing_quick_all_thred,comparing_quick_map
model2 = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=42)
selected=spa_selected
x_train100000_seleted = train_pixel[:, selected]
y_train100000=train_pixel_label
A1=np.zeros([100,18,100])
rat=0.02
EPOCH=100
# 创建tqdm进度条
pbar = tqdm(range(EPOCH), desc="翻转阈值变化训练", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [阈值: {postfix}]')
for epoch in pbar:
    threshold = epoch * 0.01
    # 更新进度条信息
    pbar.set_postfix_str(f"{threshold:.2f}")
    index_to_zeros = np.where((quality_label[:, 1] < threshold) & (quality_label[:, 0] < 1.5))[0].astype(int)
    x_train_cl = x_train100000_seleted
    y_train_cl = y_train100000
    y_train_cl[index_to_zeros]=2
    np.random.seed(7)
    model2.fit(x_train_cl, y_train_cl)

    # 计算训练集 像素级和颗粒级分类精度
    y_pred_train = model2.predict(train_pixel[:, selected])  ## 像素分类结果
    train_pred_kernel = np.zeros_like(train_img_label_selected)
    for i in range(np.shape(train_pixel)[0]):
        train_pred_kernel[train_pixel_pos[i, 0], train_pixel_pos[i, 1]] = np.array(y_pred_train[i])
    A_train1 = comparing_quick_all_thred(train_img_label_selected, train_pred_kernel,train_label_kernel_index)

    # 计算测试集 像素级和颗粒级分类精度
    y_pred_validation = model2.predict(validation_pixel[:, selected])  ## 像素分类结果
    validation_pred_kernel = np.zeros_like(validation_img_label_selected)
    for i in range(np.shape(validation_pixel)[0]):
        validation_pred_kernel[validation_pixel_pos[i, 0], validation_pixel_pos[i, 1]] = np.array(y_pred_validation[i])
    A_validation1 = comparing_quick_all_thred(validation_img_label_selected, validation_pred_kernel,validation_label_kernel_index)

    # 计算测试集 像素级和颗粒级分类精度
    y_pred_test = model2.predict(test_pixel[:, selected])  ## 像素分类结果
    test_pred_kernel = np.zeros_like(test_img_label_selected)
    for i in range(np.shape(test_pixel)[0]):
        test_pred_kernel[test_pixel_pos[i, 0], test_pixel_pos[i, 1]] = np.array(y_pred_test[i])
    A_test1 = comparing_quick_all_thred(test_img_label_selected, test_pred_kernel,test_label_kernel_index)


    x_train_acc=model2.score(train_pixel[:, selected], train_pixel_label)
    x_validation_acc=model2.score(validation_pixel[:, selected], validation_pixel_label)
    x_test_acc=model2.score(test_pixel[:, selected],test_pixel_label)
    x_train_kernel_acc=(A_train1[:,0]+A_train1[:,3])/np.sum(A_train1,axis=1)
    x_validation_kernel_acc=(A_validation1[:,0]+A_validation1[:,3])/np.sum(A_validation1,axis=1)
    x_test_kernel_acc=(A_test1[:,0]+A_test1[:,3])/np.sum(A_test1,axis=1)
    A1[:,:,epoch]=np.concatenate((A_train1,A_validation1,A_test1,
                              np.tile(np.expand_dims(x_train_acc, axis=0), (100, 1)),
                              np.tile(np.expand_dims(x_validation_acc, axis=0), (100, 1)),
                              np.tile(np.expand_dims(x_test_acc, axis=0), (100, 1)),
                              np.expand_dims(x_train_kernel_acc,axis=1),
                              np.expand_dims(x_validation_kernel_acc, axis=1),
                              np.expand_dims(x_test_kernel_acc, axis=1)),axis=1)
io.savemat('spa_rf_pixel_kernel_threshold3.mat',{'A1':A1})

# result = io.loadmat('spa_rf_pixel_kernel_threshold3.mat')
# A1 = result['A1']
aa=A1[:,17,:] #Relationship Matrix between PMP and Label Quality in Test Set
plt.figure(figsize=(8, 6))
plt.imshow(aa, cmap='viridis', aspect='auto',origin='lower')
plt.xlabel('Label Quality')
plt.ylabel('PMP')
plt.colorbar()
plt.title('2D Array Visualization')
plt.show()

####################################################################################################### 5.clean noisy labels
print("5.Identificate Moldy Peanuts using clean data ...")
optimal_threshold=0.95
optimal_PMP=0.02
index_to_zeros = np.where((quality_label[:, 1] < optimal_threshold) & (quality_label[:, 0] < 1.5))[0].astype(int)
x_train_cl = train_pixel[:, spa_selected]
y_train_cl = train_pixel_label
y_train_cl[index_to_zeros] = 2

model2=rf
np.random.seed(7)
model2.fit(x_train_cl,y_train_cl)
from sklearn.metrics import accuracy_score
# 在训练集上进行预测并计算准确率
y_pred_train = model2.predict(train_pixel[:,spa_selected])
accuracy_train = accuracy_score(train_pixel_label, y_pred_train)
print(f'Training Set Accuracy in pixel-level: {accuracy_train}')
# 在验证集上进行预测并计算准确率
y_pred_validation = model2.predict(validation_pixel[:,spa_selected])
accuracy_validation = accuracy_score(validation_pixel_label, y_pred_validation)
print(f'Validation Set Accuracy in pixel-level: {accuracy_validation}')
# 在测试集上进行预测并计算准确率
y_pred_test = model2.predict(test_pixel[:,spa_selected])
accuracy_test = accuracy_score(test_pixel_label, y_pred_test)
print(f'Test Set Accuracy in pixel-level: {accuracy_test}')


test_pred_kernel=np.zeros_like(test_img_label_selected)
for i in range(np.shape(test_pixel_label)[0]):
    test_pred_kernel[test_pixel_pos[i,0],test_pixel_pos[i,1]]=np.array(y_pred_test[i])
cv2.imwrite('test_pred.png', test_pred_kernel)

#
A_PMP= comparing_quick(test_img_label_selected, test_pred_kernel,test_label_kernel_index,optimal_PMP)
print("Confusion Matrix of test set:")
print(A_PMP)




