import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

# do not modify these unless you know what you are doing
my_output_identifier = "PECA_Net"
default_plans_identifier = "PECA_NetPlansv2.1"
default_data_identifier = 'PECA_NetData_plans_v2.1'
default_trainer = "PECA_NetTrainerV2"
default_cascade_trainer = "PECA_NetTrainerV2CascadeFullRes"

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

base = r'H:/CT/Codes_of_CT/PECA_Net/DATASET/PECA_Net_raw'  # 原数据路径
preprocessing_output_dir = r'H:/CT/Codes_of_CT/PECA_Net/DATASET/PECA_Net_preprocessed'  # 预处理后数据路径
network_training_output_dir_base = r'H:/CT/Codes_of_CT/PECA_Net/DATASET/PECA_Net_trained_models'  # 网络训练数据

if base is not None:
    PECA_Net_raw_data = join(base, "PECA_Net_raw_data")
    PECA_Net_cropped_data = join(base, "PECA_Net_cropped_data")
    maybe_mkdir_p(PECA_Net_raw_data)
    maybe_mkdir_p(PECA_Net_cropped_data)
else:
    print("PECA_Net_raw_data_base is not defined and nnU-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set this up properly.")
    PECA_Net_cropped_data = PECA_Net_raw_data = None

if preprocessing_output_dir is not None:
    maybe_mkdir_p(preprocessing_output_dir)
else:
    print("PECA_Net_preprocessed is not defined and nnU-Net can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how to set this up.")
    preprocessing_output_dir = None

if network_training_output_dir_base is not None:
    network_training_output_dir = join(network_training_output_dir_base, my_output_identifier)
    maybe_mkdir_p(network_training_output_dir)
else:
    print("RESULTS_FOLDER is not defined and PECA_Net cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information on how to set this "
          "up.")
    network_training_output_dir = None
