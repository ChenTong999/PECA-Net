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

import PECA_Net
import torch
from batchgenerators.utilities.file_and_folder_operations import *
import importlib
import pkgutil
from PECA_Net.training.network_training.PECA_NetTrainer import PECA_NetTrainer
from PECA_Net.training.network_training.PECA_NetTrainer_BTCV import PECA_NetTrainer_BTCV
from PECA_Net.training.network_training.PECA_NetTrainer_synapse import PECA_NetTrainer_synapse
from PECA_Net.paths import network_training_output_dir, preprocessing_output_dir, default_plans_identifier

def recursive_find_python_class(folder, trainer_name, current_module):
    tr = None
    # print("folder: ", folder)
    # print("trainer_name: ", trainer_name)
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        # print("ispkg: ", ispkg)
        # print("modname: ", modname)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            # print("module: ", m)
            if hasattr(m, trainer_name):
                tr = getattr(m, trainer_name)
                # print("trainer: ", tr)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class([join(folder[0], modname)], trainer_name, current_module=next_current_module)
            if tr is not None:
                break
    return tr


def restore_model(pkl_file, checkpoint=None, train=False, fp16=None,folder=None):
    """
    This is a utility function to load any PECA_Net trainer from a pkl. It will recursively search
    PECA_Net.trainig.network_training for the file that contains the trainer and instantiate it with the arguments saved in the pkl file. If checkpoint
    is specified, it will furthermore load the checkpoint file in train/test mode (as specified by train).
    The pkl file required here is the one that will be saved automatically when calling PECA_NetTrainer.save_checkpoint.
    :param pkl_file:
    :param checkpoint:
    :param train:
    :param fp16: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    
    info = load_pickle(pkl_file)
    init = info['init']
    name = info['name']
    '''
    update on 2022.6.23.
    For the model_best.model can be shared in different machines.
    '''
    task=folder.split('/')[-2]
    network = folder.split('/')[-3]
    if network == '2d':
        plans_file = join(preprocessing_output_dir, task, default_plans_identifier + "_plans_2D.pkl")
    else:
        plans_file = join(preprocessing_output_dir, task, default_plans_identifier + "_plans_3D.pkl")
    info['init'] = list(info['init'])
    info['init'][0]=plans_file
    info['init'] = tuple(info['init'])
    
    if 'nnUNet' in name:
        name=name.replace('nnUNet','PECA_Net')
    if len(init)>10:
        init=list(init)
        del init[2]
        del init[-2]
    search_in = join(PECA_Net.__path__[0], "training", "network_training")
    tr = recursive_find_python_class([search_in], name, current_module="PECA_Net.training.network_training")

    if tr is None:
        """
        Fabian only. This will trigger searching for trainer classes in other repositories as well
        """
        try:
            import meddec
            search_in = join(meddec.__path__[0], "model_training")
            tr = recursive_find_python_class([search_in], name, current_module="meddec.model_training")
        except ImportError:
            pass

    if tr is None:
        raise RuntimeError("Could not find the model trainer specified in checkpoint in PECA_Net.trainig.network_training. If it "
                           "is not located there, please move it or change the code of restore_model. Your model "
                           "trainer can be located in any directory within PECA_Net.trainig.network_training (search is recursive)."
                           "\nDebug info: \ncheckpoint file: %s\nName of trainer: %s " % (checkpoint, name))
    assert issubclass(tr, PECA_NetTrainer) or issubclass(tr, PECA_NetTrainer_synapse) or issubclass(tr, PECA_NetTrainer_BTCV), "The network trainer was found but is not a subclass of PECA_NetTrainer.Please make it so!"

    # this is now deprecated
    """if len(init) == 7:
        print("warning: this model seems to have been saved with a previous version of PECA_Net. Attempting to load it "
              "anyways. Expect the unexpected.")
        print("manually editing init args...")
        init = [init[i] for i in range(len(init)) if i != 2]"""

    # ToDo Fabian make saves use kwargs, please...

    trainer = tr(*init)

    # We can hack fp16 overwriting into the trainer without changing the init arguments because nothing happens with
    # fp16 in the init, it just saves it to a member variable
    if fp16 is not None:
        trainer.fp16 = fp16

    trainer.process_plans(info['plans'])
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint, train)
    return trainer


def load_best_model_for_inference(folder):
    checkpoint = join(folder, "model_best.model")
    pkl_file = checkpoint + ".pkl"
    return restore_model(pkl_file, checkpoint, False)


def load_model_and_checkpoint_files(folder, folds=None, mixed_precision=None, checkpoint_name="model_best"):
    """
    used for if you need to ensemble the five models of a cross-validation. This will restore the model from the
    checkpoint in fold 0, load all parameters of the five folds in ram and return both. This will allow for fast
    switching between parameters (as opposed to loading them form disk each time).

    This is best used for inference and test prediction
    :param folder:
    :param folds:
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    if isinstance(folds, str):
        folds = [join(folder, "all")]
        assert isdir(folds[0]), "no output folder for fold %s found" % folds
    elif isinstance(folds, (list, tuple)):
        if len(folds) == 1 and folds[0] == "all":
            folds = [join(folder, "all")]
        else:
            folds = [join(folder, "fold_%d" % i) for i in folds]
        assert all([isdir(i) for i in folds]), "list of folds specified but not all output folders are present"
    elif isinstance(folds, int):
        folds = [join(folder, "fold_%d" % folds)]
        assert all([isdir(i) for i in folds]), "output folder missing for fold %d" % folds
    elif folds is None:
        print("folds is None so we will automatically look for output folders (not using \'all\'!)")
        folds = subfolders(folder, prefix="fold")
        print("found the following folds: ", folds)
    else:
        raise ValueError("Unknown value for folds. Type: %s. Expected: list of int, int, str or None", str(type(folds)))

    trainer = restore_model(join(folds[0], "%s.model.pkl" % checkpoint_name), fp16=mixed_precision,folder=folder)
    trainer.output_folder = folder
    trainer.output_folder_base = folder
    trainer.update_fold(0)
    trainer.initialize(False)
    all_best_model_files = [join(i, "%s.model" % checkpoint_name) for i in folds]
    print("using the following model files: ", all_best_model_files)
    all_params = [torch.load(i, map_location=torch.device('cpu')) for i in all_best_model_files]
    return trainer, all_params


if __name__ == "__main__":
    pkl = "/home/fabian/PhD/results/PECA_NetV2/PECA_NetV2_3D_fullres/Task004_Hippocampus/fold0/model_best.model.pkl"
    checkpoint = pkl[:-4]
    train = False
    trainer = restore_model(pkl, checkpoint, train)
