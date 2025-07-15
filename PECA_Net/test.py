import pickle

with open(r'H:\CT\Codes_of_CT\PECA_Net\DATASET\PECA_Net_preprocessed\Task001_ACDC\PECA_NetPlansv2.1_plans_3D.pkl', 'rb') as f:
    loaded_dta = pickle.load(f)
print(loaded_dta)
