#
from unimol_tools import MolTrain_ep, MolPredict_ep
from rdkit import Chem
import pandas as pd
import numpy as np
import torch
import multiprocessing
import os
import pickle
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors


#SMILES standardization
def norm_smiles(raw_smiles):
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is not None:  # Ensure SMILES validity
        new_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return new_smiles
    else:
        return raw_smiles


#cross-linked structure SMILES
def link_smiles_func(combo0, combo1, catalyst_mark):

    #Case 1
    if combo0 == 'C' or combo1 == 'C':
        new_smiles = 'C'
        new_smile_mark = 0
        return new_smiles, new_smile_mark

    # cheak if combo0 has epoxy group
    mol0 = Chem.MolFromSmiles(combo0)
    if mol0 is None:
        new_smiles = 'C'
        new_smile_mark = 0
        return new_smiles, new_smile_mark

    epoxide_smarts = Chem.MolFromSmarts('[OX2]1[CX4][CX4]1')
    has_epoxide = mol0.HasSubstructMatch(epoxide_smarts)

    # cheak if combo1 has these 6 specific groups
    mol1 = Chem.MolFromSmiles(combo1)
    if mol1 is None:
        new_smiles = 'C'
        new_smile_mark = 0
        return new_smiles, new_smile_mark

    # Define SMARTS patterns for 6 functional groups
    smarts_patterns = {
        'NH': Chem.MolFromSmarts('[NX3;H1,H2]'),  # -NH- or -NH2
        'carboxyl': Chem.MolFromSmarts('[CX3](=O)[OX2H1]'),  # -COOH
        'hydroxyl': Chem.MolFromSmarts('[OX2H]'),  # -OH（Including phenolic hydroxyl groups, but excluding the -OH in carboxyl groups）
        'thiol': Chem.MolFromSmarts('[SX2H]'),  # -SH
        'cyano': Chem.MolFromSmarts('[NX1]#[CX1]'),  # -CN
        'anhydride': Chem.MolFromSmarts('[CX3](=[OX1])[OX2][CX3](=[OX1])')  # anhydride
    }

    # Count the number of each type of group
    group_counts = {}
    for group_name, smarts in smarts_patterns.items():
        if smarts is not None:
            # Deal with the special case of hydroxyl groups (excluding -OH in carboxyl groups)
            if group_name == 'hydroxyl':
                carboxyl_matches = mol1.GetSubstructMatches(smarts_patterns['carboxyl'])
                hydroxyl_matches = mol1.GetSubstructMatches(smarts)

                # Extract the index of the O atom in the carboxyl group
                carboxyl_o_indices = set()
                for match in carboxyl_matches:
                    # The third atom in carboxyl SMARTS is O
                    if len(match) >= 2:
                        carboxyl_o_indices.add(match[2])

                # Calculate the hydroxyl groups that are not in carboxyl groups
                non_carboxyl_hydroxyls = 0
                for match in hydroxyl_matches:
                    if match[0] not in carboxyl_o_indices:
                        non_carboxyl_hydroxyls += 1

                group_counts[group_name] = non_carboxyl_hydroxyls
            else:
                group_counts[group_name] = len(mol1.GetSubstructMatches(smarts))
        else:
            group_counts[group_name] = 0

    # Calculate the number of types of groups contained
    group_types = sum(1 for count in group_counts.values() if count > 0)

    # Case 2: combo0 does not contain epoxy groups, or combo1 does not contain any specified groups
    if not has_epoxide or group_types == 0:
        new_smiles = 'C'
        new_smile_mark = 2
        return new_smiles, new_smile_mark

    # Case 3: combo0 contains epoxy groups, and combo1 contains at least one specified group
    # The new_smiles are determined based on the types and numbers of groups contained in combo1
    active_groups = [group for group, count in group_counts.items() if count > 0]

    # When combo1 contains only one type of group
    if group_types == 1:
        group = active_groups[0]
        if group == 'NH':
            new_smiles = norm_smiles('CC(O)CN(C)C')
            new_smile_mark = 1
        elif group == 'carboxyl':
            new_smiles = norm_smiles('CC(=O)OCC(C)O')
            new_smile_mark = 1
        elif group == 'hydroxyl':
            new_smiles = norm_smiles('COCC(C)O')
            new_smile_mark = 1
        elif group == 'thiol':
            new_smiles = norm_smiles('CSCC(C)O')
            new_smile_mark = 1
        elif group == 'cyano':
            new_smiles = norm_smiles('CC1=NCC(C)O1')
            new_smile_mark = 1
        elif group == 'anhydride':
            new_smiles = norm_smiles('CC(=O)OCC(C)OC(C)=O')
            new_smile_mark = 1
        else:
            new_smiles = 'C'
            new_smile_mark = 2

    # When combo1 contains 2 types of groups
    elif group_types == 2:
        group1, group2 = active_groups  # 
        #print('2_groups_curing_agent')
        if {'carboxyl', 'anhydride'} == {group1, group2}:
            new_smiles = norm_smiles('CC(=O)OCC(O)CCC(COC(C)=O)OC(C)=O')
            new_smile_mark = 1
        elif {'NH', 'hydroxyl'} == {group1, group2}:
            new_smiles = norm_smiles('COCC(O)CCC(O)CN(C)C')
            new_smile_mark = 1
        elif {'NH', 'cyano'} == {group1, group2}:
            new_smiles = norm_smiles('CC1=NCC(CCC(O)CN(C)C)O1')
            new_smile_mark = 1
        elif {'carboxyl', 'hydroxyl'} == {group1, group2}:
            new_smiles = norm_smiles('COCC(O)CCC(O)COC(C)=O')
            new_smile_mark = 1
        else:
            new_smiles = 'C'
            new_smile_mark = 2

    # other situations
    else:
        new_smiles = 'C'
        new_smile_mark = 2

    return new_smiles, new_smile_mark


#Convert epoxy value to functionality and hydroxyl number
def trans_epvalue2fh(huanyangzhi_feat, smiles, jituan_feat, qiangji_feat):
    """
    The conversion of epoxy value is a function of the number of hydroxyl and epoxy groups

    parameter:
    huanyangzhi_feat: pandas Series, Epoxy value data
    smiles: pandas Series, SMILES
    jituan_feat: pandas Series, Number of epoxy groups
    qiangji_feat: pandas Series, Number of -OH

    return:
    count_exceed: int, Counting of discrepancies exceeding 6%
    real_qiangji: pandas Series, Calculated hydroxyl value
    real_epoxy: pandas Series, Calculated epoxy group value
    """

    # Check whether the input lengths are equal
    if len(huanyangzhi_feat) != len(smiles):
        raise ValueError("Length inconsistencies")

    # Initialize the counter and result list
    count_exceed = 0
    real_qiangji_list = []
    real_epoxy_list = []

    # SMARTS of epoxy groups
    epoxy_pattern = Chem.MolFromSmarts('[OX2]1[CX4][CX4]1')

    # SMARTS of -OH
    hydroxyl_pattern = Chem.MolFromSmarts('[OX2H]')

    # Traverse each element
    for i in range(len(huanyangzhi_feat)):
        # Get the value of the current element
        hyz_value = huanyangzhi_feat.iloc[i] if hasattr(huanyangzhi_feat, 'iloc') else huanyangzhi_feat[i]
        smi = smiles.iloc[i] if hasattr(smiles, 'iloc') else smiles[i]

        # Determine whether all conditions are met
        if hyz_value == 0 and smi == 'C':
            real_qiangji_list.append(0)
            real_epoxy_list.append(0)
            continue

        # Determine whether only one condition is met
        elif (hyz_value == 0 and smi != 'C'):
            mol = Chem.MolFromSmiles(smi)
            # 
            initial_epoxy = len(mol.GetSubstructMatches(epoxy_pattern))
            #print('initial_epoxy', initial_epoxy)
            # 
            initial_hydroxyl = len(mol.GetSubstructMatches(hydroxyl_pattern))
            if initial_epoxy == 0:
                real_epoxy_list.append(0)
                real_qiangji_list.append(initial_hydroxyl)
            else:
                raise ValueError(f"The data in row {i} is inconsistent: huanyangzhi_feat={hyz_value}, smiles='{smi}'")

        elif (hyz_value != 0 and smi == 'C'):
            raise ValueError(f"The data in row {i} is inconsistent: huanyangzhi_feat={hyz_value}, smiles='{smi}'")

        else:
            # SMILES
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"The SMILES string on line {i} is invalid: '{smi}'")

            # Calculate initial molecular weight
            initial_mw = Descriptors.MolWt(mol)

            # Calculate the initial number of epoxy groups
            initial_epoxy = len(mol.GetSubstructMatches(epoxy_pattern))
            #print('initial_epoxy', initial_epoxy)

            # Calculate the initial number of hydroxyl groups
            initial_hydroxyl = len(mol.GetSubstructMatches(hydroxyl_pattern))

            # 
            ideal_epoxy_value = (initial_epoxy / initial_mw) * 100

            # 
            n = 0
            mw_history = [initial_mw]
            epoxy_history = [initial_epoxy]
            hydroxyl_history = [initial_hydroxyl]

            while ideal_epoxy_value > hyz_value:
                n += 1
                # Update the molecular weight, number of epoxy groups, and number of hydroxyl groups
                new_mw = mw_history[-1] + initial_mw - 56.07
                new_epoxy = epoxy_history[-1] + initial_epoxy - 2
                new_hydroxyl = hydroxyl_history[-1] + initial_hydroxyl + 1

                # 
                mw_history.append(new_mw)
                epoxy_history.append(new_epoxy)
                hydroxyl_history.append(new_hydroxyl)

                # 
                ideal_epoxy_value = (new_epoxy / new_mw) * 100

                # Prevent infinite loops
                if n > 50:
                    break

            # cal X
            if n == 0:
                real_epoxy_val = initial_epoxy
                real_hydroxyl_val = initial_hydroxyl
            else:
                # 
                epoxy_n = epoxy_history[-1]
                epoxy_n1 = epoxy_history[-2]
                mw_n = mw_history[-1]
                mw_n1 = mw_history[-2]

                # solve: (epoxy_n*X + epoxy_n1*(1-X)) / (mw_n*X + mw_n1*(1-X)) = hyz_value/100
                # change to: A*X + B = 0
                A = epoxy_n - epoxy_n1 - (hyz_value / 100) * (mw_n - mw_n1)
                B = epoxy_n1 - (hyz_value / 100) * mw_n1

                if abs(A) < 1e-10:
                    # 
                    X = 0.5
                    raise ValueError(f"There is an error in the data on line {i}: '{i}'")
                else:
                    X = -B / A

                # make sure X in 0~1
                X = max(0, min(1, X))

                # Calculate the actual epoxy group and hydroxyl group values
                hydroxyl_n = hydroxyl_history[-1]
                hydroxyl_n1 = hydroxyl_history[-2]

                real_epoxy_val = epoxy_n * X + epoxy_n1 * (1 - X)
                real_hydroxyl_val = hydroxyl_n * X + hydroxyl_n1 * (1 - X)

            # 
            jituan_val = jituan_feat.iloc[i] if hasattr(jituan_feat, 'iloc') else jituan_feat[i]
            qiangji_val = qiangji_feat.iloc[i] if hasattr(qiangji_feat, 'iloc') else qiangji_feat[i]

            # Calculate relative error
            if jituan_val != 0:
                epoxy_error = abs(real_epoxy_val - jituan_val) / jituan_val
            else:
                epoxy_error = abs(real_epoxy_val)  # 

            if qiangji_val != 0:
                hydroxyl_error = abs(real_hydroxyl_val - qiangji_val) / qiangji_val
            else:
                hydroxyl_error = abs(real_hydroxyl_val)  #

            # If the error exceeds 6%, increment the count by 1
            if epoxy_error > 0.06 or hydroxyl_error > 0.06:
                count_exceed += 1
                #print(initial_mw, initial_epoxy, initial_hydroxyl, ideal_epoxy_value, mw_history, epoxy_history, hydroxyl_history, A, B, hyz_value)   ###!
                print(f"The data in row {i+2} is inconsistent", epoxy_error, real_epoxy_val, jituan_val, hydroxyl_error, real_hydroxyl_val, qiangji_val, smi)
                print(f"Correct data on line {i+2} is ", real_epoxy_val, real_hydroxyl_val)###excel行号

            # 
            real_qiangji_list.append(real_hydroxyl_val)
            real_epoxy_list.append(real_epoxy_val)

    # transform to Series
    if isinstance(qiangji_feat, pd.Series):
        real_qiangji = pd.Series(real_qiangji_list, index=qiangji_feat.index)
        real_epoxy = pd.Series(real_epoxy_list, index=jituan_feat.index)
    else:
        real_qiangji = np.array(real_qiangji_list)
        real_epoxy = np.array(real_epoxy_list)

    # check len
    if len(real_qiangji) != len(qiangji_feat) or len(real_epoxy) != len(jituan_feat):
        raise ValueError("The calculation results do not match the length of the input data")

    return count_exceed

'''
def trans_ca2fh(smiles, jituan_feat, qiangji_feat): #Not in use yet
    # 
    real_qiangji_list = []
    real_jituan_list = []
    for i in range(len(smiles)):
        smi = smiles.iloc[i] if hasattr(smiles, 'iloc') else smiles[i]
        # SMILES
        mol1 = Chem.MolFromSmiles(smi)
        if mol1 is None:
            raise ValueError(f"第{i}行的SMILES字符串无效: '{smi}'")

        # 
        hydroxyl_pattern = Chem.MolFromSmarts('[OX2H]')
        # 
        initial_hydroxyl = len(mol1.GetSubstructMatches(hydroxyl_pattern))
        real_hydroxyl_val = initial_hydroxyl
        # 
        real_qiangji_list.append(real_hydroxyl_val)

        # 
        smarts_patterns = {
            'NH': Chem.MolFromSmarts('[NX3;H1,H2]'),  # -NH- or -NH2
            'carboxyl': Chem.MolFromSmarts('[CX3](=O)[OX2H1]'),  # -COOH
            'hydroxyl': Chem.MolFromSmarts('[OX2H]'),  # -OH（don't include -OH in -COOH）
            'thiol': Chem.MolFromSmarts('[SX2H]'),  # -SH
            'cyano': Chem.MolFromSmarts('[NX1]#[CX1]'),  # -CN
            'anhydride': Chem.MolFromSmarts('[CX3](=[OX1])[OX2][CX3](=[OX1])')  # anhydride
        }

        # 
        group_counts = {}
        for group_name, smarts in smarts_patterns.items():
            if smarts is not None:
                # 
                if group_name == 'hydroxyl':
                    carboxyl_matches = mol1.GetSubstructMatches(smarts_patterns['carboxyl'])
                    hydroxyl_matches = mol1.GetSubstructMatches(smarts)

                    # 
                    carboxyl_o_indices = set()
                    for match in carboxyl_matches:
                        # 
                        if len(match) >= 2:
                            carboxyl_o_indices.add(match[2])

                    # 
                    non_carboxyl_hydroxyls = 0
                    for match in hydroxyl_matches:
                        if match[0] not in carboxyl_o_indices:
                            non_carboxyl_hydroxyls += 1

                    group_counts[group_name] = non_carboxyl_hydroxyls
                else:
                    group_counts[group_name] = len(mol1.GetSubstructMatches(smarts))
            else:
                group_counts[group_name] = 0

        total_groups = sum(group_counts.values())
        real_jituan_val = total_groups
        # 
        real_jituan_list.append(real_jituan_val)

    # 
    if isinstance(qiangji_feat, pd.Series):
        real_qiangji = pd.Series(real_qiangji_list, index=qiangji_feat.index)
        real_jituan = pd.Series(real_jituan_list, index=jituan_feat.index)
    else:
        real_qiangji = np.array(real_qiangji_list)
        real_jituan = np.array(real_jituan_list)

    # 
    if len(real_qiangji) != len(qiangji_feat) or len(real_jituan) != len(jituan_feat):
        raise ValueError("计算结果与输入数据长度不一致")

    return real_qiangji, real_jituan
'''

# EP read data
def read_data_EP(filename, data_col_names, Data):
    tcol = 'Tg_C'                                   ###change the name Column name of target label
    gcol = 'catalyst 0 no 1 yes'
    df = pd.read_excel(filename, sheet_name='Sheet1')
    print(f'Data Length: {len(df)}')
    if len(data_col_names) == 1:
        for col in data_col_names:
            df = df.dropna(subset=[col])
    smiles_col_names = ['monomer1', 'monomer2', 'monomer3', 'curing agent1', 'curing agent2', 'curing agent3']
    for col in smiles_col_names:
        df[col].fillna('C', inplace=True)  # Replace missing values with C
        df[col] = df[col].apply(norm_smiles)  # Standardized SMILES

    mol_col_names = [d for d in df.keys() if 'mol ratio' in d]
    assert len(mol_col_names) == len(smiles_col_names) #check len
    for col in mol_col_names:
        df[col].fillna(0, inplace=True) #Fill missing values with 0

    # ============ Verify whether the sum of molar ratios is close to 100%============
    # cal sum of mol
    mol_ratio_sums = df[mol_col_names].sum(axis=1)

    # Define the permissible range (100 ± 1.5%)
    lower_bound = 100 - 1.5
    upper_bound = 100 + 1.5

    # Find the rows that do not meet the criteria
    invalid_rows = mol_ratio_sums[(mol_ratio_sums < lower_bound) | (mol_ratio_sums > upper_bound)]

    # 
    if len(invalid_rows) > 0:
        print(f"Warning: The sum of the Mol ratios for {len(invalid_rows)} rows does not approximate 100%.")
        print(f"Invalid row numbers: {invalid_rows.index.tolist()}")

        # 
        print("\nDetailed data:")
        for idx, value in invalid_rows.items():
            print(f"  Line {idx + 2}: Ratio = {value:.2f}% (Range: {lower_bound:.1f}% - {upper_bound:.1f}%)")
    else:
        print("Verified: The sum of the molar ratios for all rows falls within the range of 100 ± 1.5%.")
    #==================================================================================

    jituan_col_names = [d for d in df.keys() if 'functionality' in d]
    assert len(jituan_col_names) == len(smiles_col_names)
    for col in jituan_col_names:
        df[col].fillna(0, inplace=True)

    qiangji_col_names = [d for d in df.keys() if 'hydroxy' in d]
    assert len(qiangji_col_names) == len(smiles_col_names)
    for col in qiangji_col_names:
        df[col].fillna(0, inplace=True)

    huanyangzhi_col_names = [d for d in df.keys() if 'epoxy value' in d]
    #assert len(huanyangzhi_col_names) == len(smiles_col_names)
    for col in huanyangzhi_col_names:
        df[col].fillna(0, inplace=True)

    nnn = 0
    for idx, dcol in enumerate(smiles_col_names): #
        smiles = df[dcol] #
        smile_mark = (smiles != 'C').astype(int)  # 

        mol_feat = df[mol_col_names[idx]]
        jituan_feat = df[jituan_col_names[idx]]
        qiangji_feat = df[qiangji_col_names[idx]]

        global_feat = df[gcol]

        if len(smiles) != len(df[tcol]):
            raise RuntimeError("The length of the SMILES column does not match that of the target column. Please check the data logic.")


        if idx < 3:
            huanyangzhi_feat = df[huanyangzhi_col_names[idx]]
            nn = trans_epvalue2fh(huanyangzhi_feat, smiles, jituan_feat, qiangji_feat)  #It is used to verify the reliability of the data in the dataset
            nnn = nnn + nn
            print(nnn)
            if nnn > 10: ###
                raise ValueError("There is an error in the calculation of the number of hydroxyl groups in the epoxy group.")
        '''
        if idx >3:
            trans_ca2fh(smiles, jituan_feat, qiangji_feat)
        '''

        feat = np.stack([mol_feat, jituan_feat, qiangji_feat, global_feat], axis=1) #
        Data.append({
            'SMILES': smiles,
            'target': df[tcol],
            'feat': feat,
            'smile_mark': smile_mark   #Flag bit. 0 represents not used, 1 represents used, and 2 represents used but without edge features
        })
        #

    return None

#Information on cross-linked structure
def link_smiles_data(Data, filename):
    """
    Process SMILES data and generate new combinations
    parameter:
    Data: A list containing 6 dictionaries
    link_smiles_func: Function for processing SMILES pairs
    filename: Excel filename
    """
    # 
    df = pd.read_excel(filename, sheet_name='Sheet1')
    tcol = 'Tg_C'                                       ###change the name Column name of target label
    gcol = 'catalyst 0 no 1 yes'

    # 
    smiles_columns = [data_dict['SMILES'] for data_dict in Data]


    if len(smiles_columns) != 6:
        raise ValueError("The number of smiles is incorrect; it is not 6.")

    # 
    all_new_smiles = [[] for _ in range(9)]  # 9 columns of SMILES
    all_new_smile_marks = [[] for _ in range(9)]  # 9 columns ofsmile_mark

    num_rows = len(smiles_columns[0])
    catalyst_marks = df[gcol]
    # 
    for i in range(len(smiles_columns[0])):
        catalyst_mark = catalyst_marks.iloc[i]
        # 
        smiles_row = [smiles.iloc[i] for smiles in smiles_columns]

        # 
        group1 = smiles_row[:3]  # smile1, smile2, smile3
        group2 = smiles_row[3:]  # smile4, smile5, smile6

        # 
        combinations = []
        for s1 in group1:
            for s2 in group2:
                combinations.append((s1, s2))

        # 
        for combo_idx, combo in enumerate(combinations):
            # 
            new_smiles, new_smile_mark = link_smiles_func(combo[0], combo[1], catalyst_mark)
            # 
            all_new_smiles[combo_idx].append(new_smiles)
            all_new_smile_marks[combo_idx].append(new_smile_mark)

    # 
    for col_idx in range(9):
        # 
        smiles_col = np.array(all_new_smiles[col_idx])
        mark_col = np.array(all_new_smile_marks[col_idx])

        # 
        feat_col = np.zeros((num_rows, 4))

        Data.append({
            'SMILES': smiles_col,
            'target': df[tcol],
            'feat': feat_col,
            'smile_mark': mark_col
        })

    return None


#
def prepare_data (Data):
    #
    print("Training for task-specific adaptive")
    #
    dataset_filename = "./train_data/dataset/Tgnew.xlsx" ###change the file name
    property_map_mal = {
        '1': 'Tg_C',
        '2': 'Td5%',
        '3': 'TS',
        '4': 'epsilon_e',
        '5': 'tan_delta',
        '6': 'Eb',
    }
    train_property = []
    cate3 = ""
    while True:
        train_property_input = input("property (1:Glass transition temperature Tg, 2:5%wt thermal decomposition temperature Td5%, 3:Tensile strength TS, 4:epsilon e, 5:tan delta, 6:electrical breakdown strength Eb: ")
        if train_property_input in property_map_mal.keys():
            cate3 = property_map_mal[train_property_input]
            train_property.append(property_map_mal[train_property_input])
            break
        print("Error: Invalid input. Please re-enter a number between 1-6! (Single choice)")
    #
    read_data_EP(dataset_filename, train_property, Data)
    link_smiles_data(Data, dataset_filename)

    return cate3


def main(custom_data, cate3):
    #print(custom_data[0]['target'])
    print("start training")
    clf3 = MolTrain_ep(task='regression',  # only this model
                       data_type='molecule',
                       epochs=100, ###
                       learning_rate=4e-5,  ###
                       batch_size=4,
                       early_stopping=20,   ###
                       metrics='mae',
                       split='random',
                       save_path=f'./exp/{cate3}',
                       max_norm=5.0,
                       model_name='unimolv2',  # This project uses a single GPU for training
                       model_size='164m',  # 84m 164m 310m ...
                       load_model_dir=f'./domain_weight/{cate3}'
                       )
    clf3.fit(custom_data)

if __name__ == "__main__":
    data = []
    cate3 = prepare_data(data)
    main(data, cate3)

