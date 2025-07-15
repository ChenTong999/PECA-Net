#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from PECA_Net.paths import network_training_output_dir

if __name__ == "__main__":
    # run collect_all_fold0_results_and_summarize_in_one_csv.py first
    summary_files_dir = join(network_training_output_dir, "summary_jsons_fold0_new")
    output_file = join(network_training_output_dir, "summary.csv")

    folds = (0, )
    folds_str = ""
    for f in folds:
        folds_str += str(f)

    plans = "PECA_NetPlans"

    overwrite_plans = {
        'PECA_NetTrainerV2_2': ["PECA_NetPlans", "PECA_NetPlansisoPatchesInVoxels"], # r
        'PECA_NetTrainerV2': ["PECA_NetPlansnonCT", "PECA_NetPlansCT2", "PECA_NetPlansallConv3x3",
                            "PECA_NetPlansfixedisoPatchesInVoxels", "PECA_NetPlanstargetSpacingForAnisoAxis",
                            "PECA_NetPlanspoolBasedOnSpacing", "PECA_NetPlansfixedisoPatchesInmm", "PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_warmup': ["PECA_NetPlans", "PECA_NetPlansv2.1", "PECA_NetPlansv2.1_big", "PECA_NetPlansv2.1_verybig"],
        'PECA_NetTrainerV2_cycleAtEnd': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_cycleAtEnd2': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_reduceMomentumDuringTraining': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_graduallyTransitionFromCEToDice': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_independentScalePerAxis': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_Mish': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_Ranger_lr3en4': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_fp32': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_GN': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_momentum098': ["PECA_NetPlans", "PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_momentum09': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_DP': ["PECA_NetPlansv2.1_verybig"],
        'PECA_NetTrainerV2_DDP': ["PECA_NetPlansv2.1_verybig"],
        'PECA_NetTrainerV2_FRN': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_resample33': ["PECA_NetPlansv2.3"],
        'PECA_NetTrainerV2_O2': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_ResencUNet': ["PECA_NetPlans_FabiansResUNet_v2.1"],
        'PECA_NetTrainerV2_DA2': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_allConv3x3': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_ForceBD': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_ForceSD': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_LReLU_slope_2en1': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_lReLU_convReLUIN': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_ReLU': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_ReLU_biasInSegOutput': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_ReLU_convReLUIN': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_lReLU_biasInSegOutput': ["PECA_NetPlansv2.1"],
        #'PECA_NetTrainerV2_Loss_MCC': ["PECA_NetPlansv2.1"],
        #'PECA_NetTrainerV2_Loss_MCCnoBG': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_Loss_DicewithBG': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_Loss_Dice_LR1en3': ["PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_Loss_Dice': ["PECA_NetPlans", "PECA_NetPlansv2.1"],
        'PECA_NetTrainerV2_Loss_DicewithBG_LR1en3': ["PECA_NetPlansv2.1"],
        # 'PECA_NetTrainerV2_fp32': ["PECA_NetPlansv2.1"],
        # 'PECA_NetTrainerV2_fp32': ["PECA_NetPlansv2.1"],
        # 'PECA_NetTrainerV2_fp32': ["PECA_NetPlansv2.1"],
        # 'PECA_NetTrainerV2_fp32': ["PECA_NetPlansv2.1"],
        # 'PECA_NetTrainerV2_fp32': ["PECA_NetPlansv2.1"],

    }

    trainers = ['PECA_NetTrainer'] + ['PECA_NetTrainerNewCandidate%d' % i for i in range(1, 28)] + [
        'PECA_NetTrainerNewCandidate24_2',
        'PECA_NetTrainerNewCandidate24_3',
        'PECA_NetTrainerNewCandidate26_2',
        'PECA_NetTrainerNewCandidate27_2',
        'PECA_NetTrainerNewCandidate23_always3DDA',
        'PECA_NetTrainerNewCandidate23_corrInit',
        'PECA_NetTrainerNewCandidate23_noOversampling',
        'PECA_NetTrainerNewCandidate23_softDS',
        'PECA_NetTrainerNewCandidate23_softDS2',
        'PECA_NetTrainerNewCandidate23_softDS3',
        'PECA_NetTrainerNewCandidate23_softDS4',
        'PECA_NetTrainerNewCandidate23_2_fp16',
        'PECA_NetTrainerNewCandidate23_2',
        'PECA_NetTrainerVer2',
        'PECA_NetTrainerV2_2',
        'PECA_NetTrainerV2_3',
        'PECA_NetTrainerV2_3_CE_GDL',
        'PECA_NetTrainerV2_3_dcTopk10',
        'PECA_NetTrainerV2_3_dcTopk20',
        'PECA_NetTrainerV2_3_fp16',
        'PECA_NetTrainerV2_3_softDS4',
        'PECA_NetTrainerV2_3_softDS4_clean',
        'PECA_NetTrainerV2_3_softDS4_clean_improvedDA',
        'PECA_NetTrainerV2_3_softDS4_clean_improvedDA_newElDef',
        'PECA_NetTrainerV2_3_softDS4_radam',
        'PECA_NetTrainerV2_3_softDS4_radam_lowerLR',

        'PECA_NetTrainerV2_2_schedule',
        'PECA_NetTrainerV2_2_schedule2',
        'PECA_NetTrainerV2_2_clean',
        'PECA_NetTrainerV2_2_clean_improvedDA_newElDef',

        'PECA_NetTrainerV2_2_fixes', # running
        'PECA_NetTrainerV2_BN', # running
        'PECA_NetTrainerV2_noDeepSupervision', # running
        'PECA_NetTrainerV2_softDeepSupervision', # running
        'PECA_NetTrainerV2_noDataAugmentation', # running
        'PECA_NetTrainerV2_Loss_CE', # running
        'PECA_NetTrainerV2_Loss_CEGDL',
        'PECA_NetTrainerV2_Loss_Dice',
        'PECA_NetTrainerV2_Loss_DiceTopK10',
        'PECA_NetTrainerV2_Loss_TopK10',
        'PECA_NetTrainerV2_Adam', # running
        'PECA_NetTrainerV2_Adam_PECA_NetTrainerlr', # running
        'PECA_NetTrainerV2_SGD_ReduceOnPlateau', # running
        'PECA_NetTrainerV2_SGD_lr1en1', # running
        'PECA_NetTrainerV2_SGD_lr1en3', # running
        'PECA_NetTrainerV2_fixedNonlin', # running
        'PECA_NetTrainerV2_GeLU', # running
        'PECA_NetTrainerV2_3ConvPerStage',
        'PECA_NetTrainerV2_NoNormalization',
        'PECA_NetTrainerV2_Adam_ReduceOnPlateau',
        'PECA_NetTrainerV2_fp16',
        'PECA_NetTrainerV2', # see overwrite_plans
        'PECA_NetTrainerV2_noMirroring',
        'PECA_NetTrainerV2_momentum09',
        'PECA_NetTrainerV2_momentum095',
        'PECA_NetTrainerV2_momentum098',
        'PECA_NetTrainerV2_warmup',
        'PECA_NetTrainerV2_Loss_Dice_LR1en3',
        'PECA_NetTrainerV2_NoNormalization_lr1en3',
        'PECA_NetTrainerV2_Loss_Dice_squared',
        'PECA_NetTrainerV2_newElDef',
        'PECA_NetTrainerV2_fp32',
        'PECA_NetTrainerV2_cycleAtEnd',
        'PECA_NetTrainerV2_reduceMomentumDuringTraining',
        'PECA_NetTrainerV2_graduallyTransitionFromCEToDice',
        'PECA_NetTrainerV2_insaneDA',
        'PECA_NetTrainerV2_independentScalePerAxis',
        'PECA_NetTrainerV2_Mish',
        'PECA_NetTrainerV2_Ranger_lr3en4',
        'PECA_NetTrainerV2_cycleAtEnd2',
        'PECA_NetTrainerV2_GN',
        'PECA_NetTrainerV2_DP',
        'PECA_NetTrainerV2_FRN',
        'PECA_NetTrainerV2_resample33',
        'PECA_NetTrainerV2_O2',
        'PECA_NetTrainerV2_ResencUNet',
        'PECA_NetTrainerV2_DA2',
        'PECA_NetTrainerV2_allConv3x3',
        'PECA_NetTrainerV2_ForceBD',
        'PECA_NetTrainerV2_ForceSD',
        'PECA_NetTrainerV2_ReLU',
        'PECA_NetTrainerV2_LReLU_slope_2en1',
        'PECA_NetTrainerV2_lReLU_convReLUIN',
        'PECA_NetTrainerV2_ReLU_biasInSegOutput',
        'PECA_NetTrainerV2_ReLU_convReLUIN',
        'PECA_NetTrainerV2_lReLU_biasInSegOutput',
        'PECA_NetTrainerV2_Loss_DicewithBG_LR1en3',
        #'PECA_NetTrainerV2_Loss_MCCnoBG',
        'PECA_NetTrainerV2_Loss_DicewithBG',
        # 'PECA_NetTrainerV2_Loss_Dice_LR1en3',
        # 'PECA_NetTrainerV2_Ranger_lr3en4',
        # 'PECA_NetTrainerV2_Ranger_lr3en4',
        # 'PECA_NetTrainerV2_Ranger_lr3en4',
        # 'PECA_NetTrainerV2_Ranger_lr3en4',
        # 'PECA_NetTrainerV2_Ranger_lr3en4',
        # 'PECA_NetTrainerV2_Ranger_lr3en4',
        # 'PECA_NetTrainerV2_Ranger_lr3en4',
        # 'PECA_NetTrainerV2_Ranger_lr3en4',
        # 'PECA_NetTrainerV2_Ranger_lr3en4',
        # 'PECA_NetTrainerV2_Ranger_lr3en4',
        # 'PECA_NetTrainerV2_Ranger_lr3en4',
        # 'PECA_NetTrainerV2_Ranger_lr3en4',
        # 'PECA_NetTrainerV2_Ranger_lr3en4',
    ]

    datasets = \
        {"Task001_BrainTumour": ("3d_fullres", ),
        "Task002_Heart": ("3d_fullres",),
        #"Task024_Promise": ("3d_fullres",),
        #"Task027_ACDC": ("3d_fullres",),
        "Task003_Liver": ("3d_fullres", "3d_lowres"),
        "Task004_Hippocampus": ("3d_fullres",),
        "Task005_Prostate": ("3d_fullres",),
        "Task006_Lung": ("3d_fullres", "3d_lowres"),
        "Task007_Pancreas": ("3d_fullres", "3d_lowres"),
        "Task008_HepaticVessel": ("3d_fullres", "3d_lowres"),
        "Task009_Spleen": ("3d_fullres", "3d_lowres"),
        "Task010_Colon": ("3d_fullres", "3d_lowres"),}

    expected_validation_folder = "validation_raw"
    alternative_validation_folder = "validation"
    alternative_alternative_validation_folder = "validation_tiledTrue_doMirror_True"

    interested_in = "mean"

    result_per_dataset = {}
    for d in datasets:
        result_per_dataset[d] = {}
        for c in datasets[d]:
            result_per_dataset[d][c] = []

    valid_trainers = []
    all_trainers = []

    with open(output_file, 'w') as f:
        f.write("trainer,")
        for t in datasets.keys():
            s = t[4:7]
            for c in datasets[t]:
                s1 = s + "_" + c[3]
                f.write("%s," % s1)
        f.write("\n")

        for trainer in trainers:
            trainer_plans = [plans]
            if trainer in overwrite_plans.keys():
                trainer_plans = overwrite_plans[trainer]

            result_per_dataset_here = {}
            for d in datasets:
                result_per_dataset_here[d] = {}

            for p in trainer_plans:
                name = "%s__%s" % (trainer, p)
                all_present = True
                all_trainers.append(name)

                f.write("%s," % name)
                for dataset in datasets.keys():
                    for configuration in datasets[dataset]:
                        summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, expected_validation_folder, folds_str))
                        if not isfile(summary_file):
                            summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, alternative_validation_folder, folds_str))
                            if not isfile(summary_file):
                                summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (
                                dataset, configuration, trainer, p, alternative_alternative_validation_folder, folds_str))
                                if not isfile(summary_file):
                                    all_present = False
                                    print(name, dataset, configuration, "has missing summary file")
                        if isfile(summary_file):
                            result = load_json(summary_file)['results'][interested_in]['mean']['Dice']
                            result_per_dataset_here[dataset][configuration] = result
                            f.write("%02.4f," % result)
                        else:
                            f.write("NA,")
                            result_per_dataset_here[dataset][configuration] = 0

                f.write("\n")

                if True:
                    valid_trainers.append(name)
                    for d in datasets:
                        for c in datasets[d]:
                            result_per_dataset[d][c].append(result_per_dataset_here[d][c])

    invalid_trainers = [i for i in all_trainers if i not in valid_trainers]

    num_valid = len(valid_trainers)
    num_datasets = len(datasets.keys())
    # create an array that is trainer x dataset. If more than one configuration is there then use the best metric across the two
    all_res = np.zeros((num_valid, num_datasets))
    for j, d in enumerate(datasets.keys()):
        ks = list(result_per_dataset[d].keys())
        tmp = result_per_dataset[d][ks[0]]
        for k in ks[1:]:
            for i in range(len(tmp)):
                tmp[i] = max(tmp[i], result_per_dataset[d][k][i])
        all_res[:, j] = tmp

    ranks_arr = np.zeros_like(all_res)
    for d in range(ranks_arr.shape[1]):
        temp = np.argsort(all_res[:, d])[::-1] # inverse because we want the highest dice to be rank0
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))

        ranks_arr[:, d] = ranks

    mn = np.mean(ranks_arr, 1)
    for i in np.argsort(mn):
        print(mn[i], valid_trainers[i])

    print()
    print(valid_trainers[np.argmin(mn)])
