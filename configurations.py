import configparser

config = configparser.ConfigParser() 
config.read('configurations.ini') 
 
EXPERIMENT_NAME = config['CONFIGURATIONS']['EXPERIMENT_NAME']
DATA_DIR    = config['CONFIGURATIONS']['DATA_DIR']
BATCH_SIZE  = config.getint('CONFIGURATIONS','BATCH_SIZE')
NUM_WORKERS = config.getint('CONFIGURATIONS','NUM_WORKERS')
NUM_EPOCHS  = config.getint('CONFIGURATIONS','NUM_EPOCHS')

FOLDS  = config.getint('CONFIGURATIONS','FOLDS')
VAL_PROPORTION  = config.getfloat('CONFIGURATIONS','VAL_PROPORTION')  # out of total
#test_proportion = 1/FOLDS ... Automatic (Do not change)

OVERLAP_PERCENTAGES = eval(config['CONFIGURATIONS']['OVERLAP_PERCENTAGES']) 
THRESHOLDS = eval(config['CONFIGURATIONS']['THRESHOLDS']) 

# For function generate_test_vid output video
FPS                = config.getfloat('CONFIGURATIONS','FPS')
SPATIAL_RESOLUTION = eval(config['CONFIGURATIONS']['SPATIAL_RESOLUTION']) 
   
TEST_RESOLUTION =  eval(config['CONFIGURATIONS']['TEST_RESOLUTION']) 
