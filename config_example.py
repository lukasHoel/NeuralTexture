# =============== Basic Configurations ===========
TEXTURE_W = 640
TEXTURE_H = 480
TEXTURE_DIM = 16
USE_PYRAMID = True
VIEW_DIRECTION = False


# =============== Train Configurations ===========
DATA_DIR = '/home/lukas/Desktop/0128667499bc73c869df6b20a2d4fe26'
CHECKPOINT_DIR = ''
LOG_DIR = ''
#TRAIN_SET = ['{:04d}'.format(i) for i in range(899)]
EPOCH = 50
BATCH_SIZE = 12
CROP_W = 256
CROP_H = 256
LEARNING_RATE = 1e-3
BETAS = '0.9, 0.999'
L2_WEIGHT_DECAY = '0.01, 0.001, 0.0001, 0'
EPS = 1e-8
LOAD = None
LOAD_STEP = 0
EPOCH_PER_CHECKPOINT = 50


# =============== Test Configurations ============
TEST_LOAD = ''
TEST_DATA_DIR = ''
TEST_SET = ['{:04d}'.format(i) for i in range(10)]
SAVE_DIR = ''