#domain-specific adaptation
from unimol_tools import MolTrain, MolPredict, UniMolRepr
from rdkit import Chem
import pandas as pd
import numpy as np
import torch
import multiprocessing
import os
import pickle

#SMILES standardization
def norm_smiles(raw_smiles):
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is not None:  # Ensure that SMILES is effective
        new_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return new_smiles
    else:
        return raw_smiles

# Read the data
def read_data_pretrains(filename, data_col_names):
    df = pd.read_excel(filename, sheet_name='Sheet1')
    smiles_col = 'smiles'
    df[smiles_col] = df[smiles_col].apply(norm_smiles)  # Standardized SMILES
    target_col = data_col_names
    smiles_list = df[smiles_col]
    data_list = df[target_col]
    #print(data_list)
    return smiles_list, data_list

#
def main0 ():
    #
    print("Train for domain-specific adaptation :")

    #Read the data from .xlsx
    dataset_filename = "./dataset/organic_molecular_properties.xlsx"  ###If necessary, please change the filename 
    property_map_mal = {
        '1': ['Melting Point', 'Density', 'LogP', 'Dipole_Debye', 'ZPE_kJ/mol', 'H_correction_kJ/mol'],
        '2': ['Melting Point', 'ABE_kJ/mol', 'LogP', 'Dipole_Debye', 'ZPE_kJ/mol', 'H_correction_kJ/mol'],
        '3': ['Density', 'Melting Point', 'ABE_kJ/mol', 'LogP', 'Dipole_Debye'],
        '4': ['Density', 'Melting Point', 'LogP', 'Dipole_Debye', 'Polarizability_10-40 C2·m2·J-1', 'ZPE_kJ/mol', 'H_correction_kJ/mol'],
        '5': ['Density', 'Melting Point', 'LogP', 'Dipole_Debye', 'HOMO_eV', 'LUMO_eV', 'VIP_eV', 'VEA_eV', 'ABE_kJ/mol']
    }
    property_map_mal_all = {
        '1': 'Tg',
        '2': 'Td5%',
        '3': 'TS',
        '4': 'e_tan',
        '5': 'Eb'
    }
    train_property = []
    cate3 = ""
    while True:
        train_property_input = input("property 1:Tg, 2:Td5%, 3:TS, 4:e&tan, 5:Eb:")
        if train_property_input in property_map_mal.keys():
            cate3 = property_map_mal_all[train_property_input]
            train_property = property_map_mal[train_property_input]
            break
        print("Error: Invalid input. Please re-enter a number between 1 and 5! (Single choice)")
    #
    print(train_property)
    read_data_3_smiles, read_data_3_target = read_data_pretrains(dataset_filename, train_property)
    return read_data_3_smiles, read_data_3_target, cate3

def main3(read_data_3_smiles, read_data_3_target, cate3):
    custom_data = {'target': read_data_3_target,
                   'SMILES': read_data_3_smiles,
                   }
    print(len(custom_data['target']))
    print("start training")
    clf3 = MolTrain(task='multilabel_regression',
                    data_type='molecule',
                    epochs=10, ###
                    learning_rate=8e-5, ###
                    batch_size=8,
                    early_stopping=3,
                    metrics='mae',
                    split='random',
                    save_path=f'./exp/{cate3}',
                    max_norm=5.0,
                    model_name='unimolv2',
                    model_size='164m' #can select: 84m 164m 310m 570m 1.1B
                    )
    clf3.fit(custom_data)

if __name__ == "__main__":
    read_data_3_smiles, read_data_3_target, cate3 = main0()
    main3(read_data_3_smiles, read_data_3_target, cate3)

