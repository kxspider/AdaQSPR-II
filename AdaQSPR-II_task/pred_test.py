from unimol_tools import MolTrain_ep, MolPredict_ep
from rdkit import Chem
import pandas as pd
import numpy as np
import torch
import multiprocessing
import os
import pickle
import joblib
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

def norm_smiles(raw_smiles):
    mol = Chem.MolFromSmiles(raw_smiles)
    if mol is not None:
        new_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return new_smiles
    else:
        return raw_smiles

def link_smiles_func(combo0, combo1, catalyst_mark):

    if combo0 == 'C' or combo1 == 'C':
        new_smiles = 'C'
        new_smile_mark = 0
        return new_smiles, new_smile_mark

    mol0 = Chem.MolFromSmiles(combo0)
    if mol0 is None:
        new_smiles = 'C'
        new_smile_mark = 0
        return new_smiles, new_smile_mark

    epoxide_smarts = Chem.MolFromSmarts('[OX2]1[CX4][CX4]1')
    has_epoxide = mol0.HasSubstructMatch(epoxide_smarts)

    mol1 = Chem.MolFromSmiles(combo1)
    if mol1 is None:
        new_smiles = 'C'
        new_smile_mark = 0
        return new_smiles, new_smile_mark

    smarts_patterns = {
        'NH': Chem.MolFromSmarts('[NX3;H1,H2]'),
        'carboxyl': Chem.MolFromSmarts('[CX3](=O)[OX2H1]'),
        'hydroxyl': Chem.MolFromSmarts('[OX2H]'),
        'thiol': Chem.MolFromSmarts('[SX2H]'),
        'cyano': Chem.MolFromSmarts('[NX1]#[CX1]'),
        'anhydride': Chem.MolFromSmarts('[CX3](=[OX1])[OX2][CX3](=[OX1])')
    }

    group_counts = {}
    for group_name, smarts in smarts_patterns.items():
        if smarts is not None:
            if group_name == 'hydroxyl':
                carboxyl_matches = mol1.GetSubstructMatches(smarts_patterns['carboxyl'])
                hydroxyl_matches = mol1.GetSubstructMatches(smarts)

                carboxyl_o_indices = set()
                for match in carboxyl_matches:
                    if len(match) >= 2:
                        carboxyl_o_indices.add(match[2])

                non_carboxyl_hydroxyls = 0
                for match in hydroxyl_matches:
                    if match[0] not in carboxyl_o_indices:
                        non_carboxyl_hydroxyls += 1

                group_counts[group_name] = non_carboxyl_hydroxyls
            else:
                group_counts[group_name] = len(mol1.GetSubstructMatches(smarts))
        else:
            group_counts[group_name] = 0

    group_types = sum(1 for count in group_counts.values() if count > 0)

    if not has_epoxide or group_types == 0:
        new_smiles = 'C'
        new_smile_mark = 2
        return new_smiles, new_smile_mark

    active_groups = [group for group, count in group_counts.items() if count > 0]

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

    elif group_types == 2:
        group1, group2 = active_groups
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

    else:
        new_smiles = 'C'
        new_smile_mark = 2

    return new_smiles, new_smile_mark

def trans_epvalue2fh(epoxy_value_feat, smiles, func_group_feat, hydroxyl_feat):
    """
    Function to convert epoxy value to hydroxyl and epoxy group counts

    Parameters:
    epoxy_value_feat: pandas Series, epoxy value data
    smiles: pandas Series, SMILES string data
    func_group_feat: pandas Series, functional group data (for validating epoxy groups)
    hydroxyl_feat: pandas Series, hydroxyl data (for validating hydroxyl)

    Returns:
    count_exceed: int, count exceeding 6% difference
    real_hydroxyl: pandas Series, calculated hydroxyl values
    real_epoxy: pandas Series, calculated epoxy group values
    """

    if len(epoxy_value_feat) != len(smiles):
        raise ValueError("epoxy_value_feat and smiles have inconsistent lengths")

    count_exceed = 0
    real_hydroxyl_list = []
    real_epoxy_list = []

    epoxy_pattern = Chem.MolFromSmarts('[OX2]1[CX4][CX4]1')

    hydroxyl_pattern = Chem.MolFromSmarts('[OX2H]')

    for i in range(len(epoxy_value_feat)):
        hyz_value = epoxy_value_feat.iloc[i] if hasattr(epoxy_value_feat, 'iloc') else epoxy_value_feat[i]
        smi = smiles.iloc[i] if hasattr(smiles, 'iloc') else smiles[i]

        if hyz_value == 0 and smi == 'C':
            real_hydroxyl_list.append(0)
            real_epoxy_list.append(0)
            continue

        elif (hyz_value == 0 and smi != 'C'):
            mol = Chem.MolFromSmiles(smi)
            initial_epoxy = len(mol.GetSubstructMatches(epoxy_pattern))
            initial_hydroxyl = len(mol.GetSubstructMatches(hydroxyl_pattern))
            if initial_epoxy == 0:
                real_epoxy_list.append(0)
                real_hydroxyl_list.append(initial_hydroxyl)
            else:
                raise ValueError(f"Row {i} data inconsistent: epoxy_value_feat={hyz_value}, smiles='{smi}'")

        elif (hyz_value != 0 and smi == 'C'):
            raise ValueError(f"Row {i} data inconsistent: epoxy_value_feat={hyz_value}, smiles='{smi}'")

        else:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Row {i} invalid SMILES string: '{smi}'")

            initial_mw = Descriptors.MolWt(mol)

            initial_epoxy = len(mol.GetSubstructMatches(epoxy_pattern))

            initial_hydroxyl = len(mol.GetSubstructMatches(hydroxyl_pattern))

            ideal_epoxy_value = (initial_epoxy / initial_mw) * 100

            n = 0
            mw_history = [initial_mw]
            epoxy_history = [initial_epoxy]
            hydroxyl_history = [initial_hydroxyl]

            while ideal_epoxy_value > hyz_value:
                n += 1
                new_mw = mw_history[-1] + initial_mw - 56.07
                new_epoxy = epoxy_history[-1] + initial_epoxy - 2
                new_hydroxyl = hydroxyl_history[-1] + initial_hydroxyl + 1

                mw_history.append(new_mw)
                epoxy_history.append(new_epoxy)
                hydroxyl_history.append(new_hydroxyl)

                ideal_epoxy_value = (new_epoxy / new_mw) * 100

                if n > 50:
                    break

            if n == 0:
                real_epoxy_val = initial_epoxy
                real_hydroxyl_val = initial_hydroxyl
            else:
                epoxy_n = epoxy_history[-1]
                epoxy_n1 = epoxy_history[-2]
                mw_n = mw_history[-1]
                mw_n1 = mw_history[-2]

                A = epoxy_n - epoxy_n1 - (hyz_value / 100) * (mw_n - mw_n1)
                B = epoxy_n1 - (hyz_value / 100) * mw_n1

                if abs(A) < 1e-10:
                    X = 0.5
                    raise ValueError(f"Row {i} data error: '{i}'")
                else:
                    X = -B / A

                X = max(0, min(1, X))

                hydroxyl_n = hydroxyl_history[-1]
                hydroxyl_n1 = hydroxyl_history[-2]

                real_epoxy_val = epoxy_n * X + epoxy_n1 * (1 - X)
                real_hydroxyl_val = hydroxyl_n * X + hydroxyl_n1 * (1 - X)

            func_group_val = func_group_feat.iloc[i] if hasattr(func_group_feat, 'iloc') else func_group_feat[i]
            hydroxyl_val = hydroxyl_feat.iloc[i] if hasattr(hydroxyl_feat, 'iloc') else hydroxyl_feat[i]

            if func_group_val != 0:
                epoxy_error = abs(real_epoxy_val - func_group_val) / func_group_val
            else:
                epoxy_error = abs(real_epoxy_val)

            if hydroxyl_val != 0:
                hydroxyl_error = abs(real_hydroxyl_val - hydroxyl_val) / hydroxyl_val
            else:
                hydroxyl_error = abs(real_hydroxyl_val)

            if epoxy_error > 0.06 or hydroxyl_error > 0.06:
                count_exceed += 1
                print(f"Row {i+2} data inconsistent", epoxy_error, real_epoxy_val, func_group_val, hydroxyl_error, real_hydroxyl_val, hydroxyl_val, smi)
                print(f"Row {i+2} correct data", real_epoxy_val, real_hydroxyl_val)

            real_hydroxyl_list.append(real_hydroxyl_val)
            real_epoxy_list.append(real_epoxy_val)

    if isinstance(hydroxyl_feat, pd.Series):
        real_hydroxyl = pd.Series(real_hydroxyl_list, index=hydroxyl_feat.index)
        real_epoxy = pd.Series(real_epoxy_list, index=func_group_feat.index)
    else:
        real_hydroxyl = np.array(real_hydroxyl_list)
        real_epoxy = np.array(real_epoxy_list)

    if len(real_hydroxyl) != len(hydroxyl_feat) or len(real_epoxy) != len(func_group_feat):
        raise ValueError("Calculation results and input data have inconsistent lengths")

    return count_exceed

def trans_ca2fh(smiles, func_group_feat, hydroxyl_feat):
    real_hydroxyl_list = []
    real_func_group_list = []
    for i in range(len(smiles)):
        smi = smiles.iloc[i] if hasattr(smiles, 'iloc') else smiles[i]
        mol1 = Chem.MolFromSmiles(smi)
        if mol1 is None:
            raise ValueError(f"Row {i} invalid SMILES string: '{smi}'")

        hydroxyl_pattern = Chem.MolFromSmarts('[OX2H]')
        initial_hydroxyl = len(mol1.GetSubstructMatches(hydroxyl_pattern))
        real_hydroxyl_val = initial_hydroxyl
        real_hydroxyl_list.append(real_hydroxyl_val)

        smarts_patterns = {
            'NH': Chem.MolFromSmarts('[NX3;H1,H2]'),
            'carboxyl': Chem.MolFromSmarts('[CX3](=O)[OX2H1]'),
            'hydroxyl': Chem.MolFromSmarts('[OX2H]'),
            'thiol': Chem.MolFromSmarts('[SX2H]'),
            'cyano': Chem.MolFromSmarts('[NX1]#[CX1]'),
            'anhydride': Chem.MolFromSmarts('[CX3](=[OX1])[OX2][CX3](=[OX1])')
        }

        group_counts = {}
        for group_name, smarts in smarts_patterns.items():
            if smarts is not None:
                if group_name == 'hydroxyl':
                    carboxyl_matches = mol1.GetSubstructMatches(smarts_patterns['carboxyl'])
                    hydroxyl_matches = mol1.GetSubstructMatches(smarts)

                    carboxyl_o_indices = set()
                    for match in carboxyl_matches:
                        if len(match) >= 2:
                            carboxyl_o_indices.add(match[2])

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
        real_func_group_val = total_groups
        real_func_group_list.append(real_func_group_val)

    if isinstance(hydroxyl_feat, pd.Series):
        real_hydroxyl = pd.Series(real_hydroxyl_list, index=hydroxyl_feat.index)
        real_func_group = pd.Series(real_func_group_list, index=func_group_feat.index)
    else:
        real_hydroxyl = np.array(real_hydroxyl_list)
        real_func_group = np.array(real_func_group_list)

    if len(real_hydroxyl) != len(hydroxyl_feat) or len(real_func_group) != len(func_group_feat):
        raise ValueError("Calculation results and input data have inconsistent lengths")

    return real_hydroxyl, real_func_group

def read_data_EP(filename, data_col_names, Data):
    tcol = 'Eb'                             ###Change to the column name of the sample data column
    gcol = 'catalyst 0 no 1 yes'
    df = pd.read_excel(filename, sheet_name='Sheet1')
    print(f'Data Length: {len(df)}')
    if len(data_col_names) == 1:
        for col in data_col_names:
            df = df.dropna(subset=[col])
    smiles_col_names = ['monomer1', 'monomer2', 'monomer3', 'curing agent1', 'curing agent2', 'curing agent3']
    for col in smiles_col_names:
        df[col].fillna('C', inplace=True)
        df[col] = df[col].apply(norm_smiles)

    mol_col_names = [d for d in df.keys() if 'mol ratio' in d]
    assert len(mol_col_names) == len(smiles_col_names)
    for col in mol_col_names:
        df[col].fillna(0, inplace=True)

    mol_ratio_sums = df[mol_col_names].sum(axis=1)

    lower_bound = 100 - 1.5
    upper_bound = 100 + 1.5

    invalid_rows = mol_ratio_sums[(mol_ratio_sums < lower_bound) | (mol_ratio_sums > upper_bound)]

    if len(invalid_rows) > 0:
        print(f"Warning: Found {len(invalid_rows)} rows where molar ratio sum is not close to 100%")
        print(f"Non-compliant row indices: {invalid_rows.index.tolist()}")

        print("\nDetailed data:")
        for idx, value in invalid_rows.items():
            print(f"  Row {idx + 2}: sum of ratios = {value:.2f}% (range: {lower_bound:.1f}% - {upper_bound:.1f}%)")
    else:
        print("Validation passed: All rows have molar ratio sums within 100±1.5%")

    func_group_col_names = [d for d in df.keys() if 'functionality' in d]
    assert len(func_group_col_names) == len(smiles_col_names)
    for col in func_group_col_names:
        df[col].fillna(0, inplace=True)

    hydroxyl_col_names = [d for d in df.keys() if 'hydroxy' in d]
    assert len(hydroxyl_col_names) == len(smiles_col_names)
    for col in hydroxyl_col_names:
        df[col].fillna(0, inplace=True)

    nnn = 0
    for idx, dcol in enumerate(smiles_col_names):
        smiles = df[dcol]
        smile_mark = (smiles != 'C').astype(int)

        mol_feat = df[mol_col_names[idx]]
        func_group_feat = df[func_group_col_names[idx]]
        hydroxyl_feat = df[hydroxyl_col_names[idx]]

        global_feat = df[gcol]

        feat = np.stack([mol_feat, func_group_feat, hydroxyl_feat, global_feat], axis=1)
        Data.append({
            'SMILES': smiles,
            'target': df[tcol],
            'feat': feat,
            'smile_mark': smile_mark
        })

    return None

def link_smiles_data(Data, filename):
    df = pd.read_excel(filename, sheet_name='Sheet1')
    tcol = 'Eb'                               ###Change to the column name of the sample data column
    gcol = 'catalyst 0 no 1 yes'

    smiles_columns = [data_dict['SMILES'] for data_dict in Data]

    if len(smiles_columns) != 6:
        raise ValueError("Number of smiles is incorrect, not 6")

    all_new_smiles = [[] for _ in range(9)]
    all_new_smile_marks = [[] for _ in range(9)]

    num_rows = len(smiles_columns[0])
    catalyst_marks = df[gcol]
    for i in range(len(smiles_columns[0])):
        catalyst_mark = catalyst_marks.iloc[i]
        smiles_row = [smiles.iloc[i] for smiles in smiles_columns]

        group1 = smiles_row[:3]
        group2 = smiles_row[3:]

        combinations = []
        for s1 in group1:
            for s2 in group2:
                combinations.append((s1, s2))

        for combo_idx, combo in enumerate(combinations):
            new_smiles, new_smile_mark = link_smiles_func(combo[0], combo[1], catalyst_mark)
            all_new_smiles[combo_idx].append(new_smiles)
            all_new_smile_marks[combo_idx].append(new_smile_mark)

    for col_idx in range(9):
        smiles_col = np.array(all_new_smiles[col_idx])
        mark_col = np.array(all_new_smile_marks[col_idx])

        feat_col = np.zeros((num_rows, 4))

        Data.append({
            'SMILES': smiles_col,
            'target': df[tcol],
            'feat': feat_col,
            'smile_mark': mark_col
        })

    return None

def prepare_data (Data):
    print("prediction")
    dataset_filename = "./pred_test/Eb_test.xlsx"          ###change the file name
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
        print("Error: Invalid input. Please re-enter a number between 1 and 6! (Single choice)")
    read_data_EP(dataset_filename, train_property, Data)
    link_smiles_data(Data, dataset_filename)

    return cate3

def main(custom_data, cate3):
    print("start training")
    clf3 = MolPredict_ep(load_model=f'./exp/{cate3}_f_0203')####change the weight name

    predict = clf3.predict(custom_data)
    sc = joblib.load(f'./exp/{cate3}_f_0203/target_scaler.ss')####change the path
    p_inverse = sc.inverse_transform(predict)
    print(p_inverse.max())
    dff = pd.read_excel('./pred_test/Eb_test.xlsx', sheet_name='Sheet1')#### change the filename
    dff[f'{cate3}_pred'] = np.array(list(p_inverse))

    excel_file = f'./pred_test/pred/prediction_{cate3}_test.xlsx'
    save_dir = os.path.dirname(excel_file)
    os.makedirs(save_dir, exist_ok=True)
    dff.to_excel(excel_file, index=False)

if __name__ == "__main__":
    data = []
    cate3 = prepare_data(data)
    main(data, cate3)