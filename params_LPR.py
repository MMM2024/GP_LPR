# alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" # RodoSol alphabets
alphabet = '皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学ABCDEFGHJKLMNPQRSTUVWXYZ0123456789' # CCPD alphabets


K = 8
random_sample = False
best_accuracy = 0.5
keep_ratio = False
adam = True
adadelta = False
saveInterval = 1
valInterval = 65
lr_step = 50
n_test_disp = 10
displayInterval = 1
gpu=1
tb_message=''
experiment = ''
model_type = 'LPR_SA_model' # 'LPR_model' or 'LPR_SA_model'
isSeqModel=True  # if SA is added
head=2
inner=256
isl2Norm=True

train_root = ''
train_data = ''
val_data = ''

pre_model = ''
beta1 =0.5
lr = 0.001
niter = 300

imgW = 96
imgH = 32
val_batchSize = 128
batchSize = 128


workers = 4
std = 1
mean = 0
