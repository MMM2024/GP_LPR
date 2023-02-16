#alphabet = '皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学ABCDEFGHJKLMNPQRSTUVWXYZ0123456789' # CCPD alphabets
alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" # RodoSol and AOLP alphabets


K = 8
 
std = 1
mean = 0
imgW = 96

imgH = 32

val_batchSize = 128
workers = 8
gpu = 3

model_type = 'LPR_model'
isSeqModel=True # GPM add or not
head=2
inner=256
isl2Norm=True

model_path = '/path/to/checkpoint.pth'

image_dir = ''
image_path = '/path/to/dataset'


