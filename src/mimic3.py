import logging
import os
import random
import shutil
import pandas as pd
import vaex
import pyarrow as pa
import numpy as np
import pyarrow.compute as pc
import pyarrow.feather 
# from pyhealth.datasets.base_ehr_dataset import BaseEHRDataset
from scipy.stats import entropy
from collections import Counter
from functools import partial
from pathlib import Path
from collections import defaultdict
from operator import itemgetter
from typing import Any, Tuple, List
# The dataset requires a Licence in physionet. Once it is obtained, download the dataset with the following command in the terminal:
# wget -r -N -c -np --user <your_physionet_user_name> --ask-password https://physionet.org/files/mimiciii/1.4/
# Change the path of DOWNLOAD_DIRECTORY to the path where you downloaded mimiciii

# Global class of repeated variables used in the csv files
class Config:
    PAD_TOKEN = "<PAD>"
    UNKNOWN_TOKEN = "<UNK>"
    ID_COLUMN = "_id"
    TEXT_COLUMN = "text"
    TARGET_COLUMN = "target"
    SUBJECT_ID_COLUMN = "subject_id"

class TextPreprocessor:
    def __init__(
        self,
        lower: bool = True,
        remove_special_characters_mullenbach: bool = True,
        remove_special_characters: bool = False,
        remove_digits: bool = True,
        remove_accents: bool = False,
        remove_brackets: bool = False,
        convert_danish_characters: bool = False,
    ) -> None:
        self.lower = lower
        self.remove_special_characters_mullenbach = remove_special_characters_mullenbach
        self.remove_digits = remove_digits
        self.remove_accents = remove_accents
        self.remove_special_characters = remove_special_characters
        self.remove_brackets = remove_brackets
        self.convert_danish_characters = convert_danish_characters

    def __call__(self, df: vaex.dataframe.DataFrame) -> vaex.dataframe.DataFrame:
        if self.lower:
            df[Config.TEXT_COLUMN] = df[Config.TEXT_COLUMN].str.lower()
        if self.convert_danish_characters:
            df[Config.TEXT_COLUMN] = df[Config.TEXT_COLUMN].str.replace("å", "aa")
            df[Config.TEXT_COLUMN] = df[Config.TEXT_COLUMN].str.replace("æ", "ae")
            df[Config.TEXT_COLUMN] = df[Config.TEXT_COLUMN].str.replace("ø", "oe")
        if self.remove_accents:
            df[Config.TEXT_COLUMN] = df[Config.TEXT_COLUMN].str.replace("é|è|ê", "e")
            df[Config.TEXT_COLUMN] = df[Config.TEXT_COLUMN].str.replace("á|à|â", "a")
            df[Config.TEXT_COLUMN] = df[Config.TEXT_COLUMN].str.replace("ô|ó|ò", "o")
        if self.remove_brackets:
            df[Config.TEXT_COLUMN] = df[Config.TEXT_COLUMN].str.replace(r"\[[^]]*\]", "")
        if self.remove_special_characters:
            df[Config.TEXT_COLUMN] = df[Config.TEXT_COLUMN].str.replace(r"\n|/|-", " ")
            df[Config.TEXT_COLUMN] = df[Config.TEXT_COLUMN].str.replace(r"[^a-zA-Z0-9 ]", "")
        if self.remove_special_characters_mullenbach:
            df[Config.TEXT_COLUMN] = df[Config.TEXT_COLUMN].str.replace(r"[^A-Za-z0-9]+", " ")
        if self.remove_digits:
            df[Config.TEXT_COLUMN] = df[Config.TEXT_COLUMN].str.replace(r"(\s\d+)+\s", " ")
        df[Config.TEXT_COLUMN] = df[Config.TEXT_COLUMN].str.replace(r"\s+", " ")
        df[Config.TEXT_COLUMN] = df[Config.TEXT_COLUMN].str.strip()
        return df

def reformat_code_dataframe(row: pd.DataFrame, col: str) -> pd.Series:
    """Takes a dataframe and a column name and returns a series with the column name and a list of codes.

    Example:
        Input:

                subject_id  _id     icd9_diag
        608           2   163353     V3001
        609           2   163353      V053
        610           2   163353      V290

        Output:

        icd9_diag    [V053, V290, V3001]

    Args:
        row (pd.DataFrame): Dataframe with a column of codes.
        col (str): column name of the codes.

    Returns:
        pd.Series: Series with the column name and a list of codes.
    """
    return pd.Series({col: row[col].sort_values().tolist()})


def format_code_dataframe(df: pd.DataFrame, col_in: str, col_out: str) -> pd.DataFrame:
    """Formats the code dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the codes.
        col_in (str): The name of the column containing the codes.
        col_out (str): The new name of the column containing the codes

    Returns:
        pd.DataFrame: The formatted dataframe.
    """
    df = df.rename(
        columns={
            "HADM_ID": Config.ID_COLUMN,
            "SUBJECT_ID": Config.SUBJECT_ID_COLUMN,
            "TEXT": Config.TEXT_COLUMN,
        }
    )
    df = df.sort_values([Config.SUBJECT_ID_COLUMN, Config.ID_COLUMN])
    df[col_in] = df[col_in].astype(str).str.strip()
    df = df[[Config.SUBJECT_ID_COLUMN, Config.ID_COLUMN, col_in]].rename({col_in: col_out}, axis=1)
    # remove codes that are nan
    df = df[df[col_out] != "nan"]
    return (
        df.groupby([Config.SUBJECT_ID_COLUMN, Config.ID_COLUMN], group_keys=False)
        .apply(partial(reformat_code_dataframe, col=col_out))
        .reset_index()
    )


def reformat_icd9(code: str, is_diag: bool) -> str:
    """
    Put a period in the right place because the MIMIC-3 data files exclude them.
    Generally, procedure codes have dots after the first two digits,
    while diagnosis codes have dots after the first three digits.
    """
    code = "".join(code.split("."))
    if is_diag:
        if code.startswith("E"):
            if len(code) > 4:
                return code[:4] + "." + code[4:]
        else:
            if len(code) > 3:
                return code[:3] + "." + code[3:]
    else:
        if len(code) > 2:
            return code[:2] + "." + code[2:]
    return code

def preprocess_documents(
    df: pd.DataFrame, preprocessor: TextPreprocessor
) -> pd.DataFrame:
    with vaex.cache.memory_infinite():  # pylint: disable=not-context-manager
        df = vaex.from_pandas(df)
        df = preprocessor(df)
        df["num_words"] = df.text.str.count(" ") + 1
        df["num_targets"] = df[Config.TARGET_COLUMN].apply(len)
        return df.to_pandas_df()
    


def merge_code_dataframes(code_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merges all code dataframes into a single dataframe.

    Args:
        code_dfs (list[pd.DataFrame]): List of code dataframes.

    Returns:
        pd.DataFrame: Merged code dataframe.
    """
    merged_codes = code_dfs[0]
    for code_df in code_dfs[1:]:
        merged_codes = merged_codes.merge(
            code_df, how="outer", on=[Config.SUBJECT_ID_COLUMN, Config.ID_COLUMN]
        )
    return merged_codes

def merge_report_addendum_helper_function(row: pd.DataFrame) -> pd.Series:
    """Merges the report and addendum text."""
    dout = dict()
    if len(row) == 1:
        dout["DESCRIPTION"] = row.iloc[0].DESCRIPTION
        dout[Config.TEXT_COLUMN] = row.iloc[0][Config.TEXT_COLUMN]
    else:
        # row = row.sort_values(["DESCRIPTION", "CHARTDATE"], ascending=[False, True])
        dout["DESCRIPTION"] = "+".join(row.DESCRIPTION)
        dout[Config.TEXT_COLUMN] = " ".join(row[Config.TEXT_COLUMN])
    return pd.Series(dout)

def merge_reports_addendum(mimic_notes: pd.DataFrame) -> pd.DataFrame:
    """Merges the reports and addendum into one dataframe.

    Args:
        mimic_notes (pd.DataFrame): The dataframe containing the notes from the mimiciii dataset.

    Returns:
        pd.DataFrame: The dataframe containing the discharge summaries consisting of reports and addendum.
    """
    discharge_summaries = mimic_notes[mimic_notes["CATEGORY"] == "Discharge summary"].copy()
    discharge_summaries[Config.ID_COLUMN] = discharge_summaries[Config.ID_COLUMN].astype(int)
    return (
        discharge_summaries.groupby([Config.SUBJECT_ID_COLUMN, Config.ID_COLUMN], group_keys=False)
        .apply(merge_report_addendum_helper_function)
        .reset_index()
    )


def replace_nans_with_empty_lists(
    df: pd.DataFrame, columns: list[str] = ["icd9_diag", "icd9_proc"]
) -> pd.DataFrame:
    """Replaces nans in the columns with empty lists."""
    for column in columns:
        df.loc[df[column].isnull(), column] = df.loc[df[column].isnull(), column].apply(
            lambda x: []
        )
    return df

def remove_duplicated_codes(df: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    """Remove duplicated codes in the dataframe"""
    df = df.copy()
    for col in column_names:
        df[col] = df[col].apply(lambda codes: list(set(codes)))
    return df




def prepare_discharge_summaries(mimic_notes: pd.DataFrame) -> pd.DataFrame:
    """Format the notes dataframe into the discharge summaries dataframe

    Args:
        mimic_notes (pd.DataFrame): The notes dataframe

    Returns:
        pd.DataFrame: Formatted discharge summaries dataframe
    """
    mimic_notes = mimic_notes.rename(
        columns={
            "HADM_ID": Config.ID_COLUMN,
            "SUBJECT_ID": Config.SUBJECT_ID_COLUMN,
            "TEXT": Config.TEXT_COLUMN,
        }
    )
    logging.info(f"{mimic_notes[Config.ID_COLUMN].nunique()} number of admissions")
    discharge_summaries = merge_reports_addendum(mimic_notes)
    discharge_summaries = discharge_summaries.sort_values(
        [Config.SUBJECT_ID_COLUMN, Config.ID_COLUMN]
    )

    discharge_summaries = discharge_summaries.reset_index(drop=True)
    logging.info(
        f"{discharge_summaries[Config.SUBJECT_ID_COLUMN].nunique()} subjects, {discharge_summaries[Config.ID_COLUMN].nunique()} admissions"
    )
    return discharge_summaries


def filter_codes(df: pd.DataFrame, columns: list[str], min_count: int) -> pd.DataFrame:
    """Filter the codes dataframe to only include codes that appear at least min_count times

    Args:
        df (pd.DataFrame): The codes dataframe
        col (str): The column name of the codes
        min_count (int): The minimum number of times a code must appear

    Returns:
        pd.DataFrame: The filtered codes dataframe
    """
    for col in columns:
        code_counts = Counter([code for codes in df[col] for code in codes])
        codes_to_keep = set(
            code for code, count in code_counts.items() if count >= min_count
        )
        df[col] = df[col].apply(lambda x: [code for code in x if code in codes_to_keep])
    return df


def download_and_preprocess_code_systems(code_systems: list[tuple], download_dir) -> pd.DataFrame:
    """Download and preprocess the code systems dataframe

    Args:
        code_systems (List[tuple]): The code systems to download and preprocess

    Returns:
        pd.DataFrame: The preprocessed code systems dataframe"""
    code_dfs = []
    for name, fname, col_in, col_out in code_systems:
        logging.info(f"Loading {name} codes...")
        df = pd.read_csv(
            os.path.join(download_dir, fname), dtype={"ICD9_CODE": str}
        )
        df = format_code_dataframe(df, col_in, col_out)
        df = remove_duplicated_codes(df, [col_out])
        code_dfs.append(df)

    merged_codes = merge_code_dataframes(code_dfs)
    merged_codes = replace_nans_with_empty_lists(merged_codes)
    merged_codes["icd9_diag"] = merged_codes["icd9_diag"].apply(
        lambda codes: list(map(partial(reformat_icd9, is_diag=True), codes))
    )
    merged_codes["icd9_proc"] = merged_codes["icd9_proc"].apply(
        lambda codes: list(map(partial(reformat_icd9, is_diag=False), codes))
    )
    merged_codes[Config.TARGET_COLUMN] = merged_codes["icd9_proc"] + merged_codes["icd9_diag"]
    return merged_codes

def get_code_system2code_counts(
    df: vaex.dataframe.DataFrame, code_systems: list[str]
) -> dict[str, dict[str, int]]:
    """

    Args:
        df (vaex.dataframe.DataFrame): The dataset in vaex dataframe format
        code_systems (list[str]): list of code systems to get counts for
    Returns:
        dict[str, dict[str, int]]: A dictionary with code systems as keys and a dictionary of code counts as values
    """
    code_system2code_counts = defaultdict(dict)
    for col in code_systems:
        codes = df[col].values.flatten().value_counts().to_pylist()
        code_system2code_counts[col] = {
            code["values"]: code["counts"] for code in codes
        }
    return code_system2code_counts




def iterative_stratification(
    data: list[Any], labels: list[list[str]], ratios: list[float]
):
    # Implemented by Sotiris Lamprinidis
    data = data.copy()
    labels = labels.copy()
    labels_unique = sorted(set([lbl for lbls in labels for lbl in lbls]))
    label_to_index = {lbl: i for i, lbl in enumerate(labels_unique)}
    desired_samples = [len(labels_unique) * r for r in ratios]
    desired_labels = [[0 for _ in range(len(labels_unique))] for r in ratios]
    sets = [list() for _ in range(len(ratios))]

    lc = Counter(lbl for lbls in labels for lbl in lbls)
    for i, label in enumerate(labels_unique):
        num_this = lc[label]
        for j, ratio in enumerate(ratios):
            desired_labels[j][i] = num_this * ratio

    while labels:
        label = lc.most_common()[-1][0]
        lbl = label_to_index[label]
        dataset_label = [
            (i, (x, y)) for i, (x, y) in enumerate(zip(data, labels)) if label in y
        ]
        for index, (x, y) in sorted(dataset_label, key=itemgetter(0), reverse=True):
            desired = sorted(
                enumerate([desired_labels[j][lbl] for j in range(len(ratios))]),
                key=itemgetter(1),
                reverse=True,
            )
            if desired[0][1] != desired[1][1]:
                chosen = desired[0]
            else:
                desired = sorted(
                    [
                        (i, desired_samples[i])
                        for i, _ in [x for x in desired if x[1] == desired[0][1]]
                    ],
                    key=itemgetter(1),
                    reverse=True,
                )
                if desired[0][1] != desired[1][1]:
                    chosen = desired[0]
                else:
                    chosen = random.choice(desired)
            sets[chosen[0]].append(x)
            del labels[index]
            del data[index]

            for label in y:
                l_this = label_to_index[label]
                desired_labels[chosen[0]][l_this] -= 1
                lc.subtract({label: 1})
                if lc[label] <= 0:
                    del lc[label]
            desired_samples[chosen[0]] -= 1
    return sets

def kl_divergence(all_codes: list[list[str]], split_codes: list[list[str]]) -> float:
    """Find KL divergence between the all and split set."""
    all_codes_unique = {code for codes in all_codes for code in codes}
    code2index = {code: i for i, code in enumerate(all_codes_unique)}
    all_counts = np.zeros(len(code2index))
    split_counts = np.zeros(len(code2index))
    for codes in all_codes:
        for code in codes:
            all_counts[code2index[code]] += 1
    for codes in split_codes:
        for code in codes:
            split_counts[code2index[code]] += 1

    all_counts = all_counts / np.sum(all_counts)
    split_counts = split_counts / np.sum(split_counts)

    return entropy(split_counts, qk=all_counts)


def labels_not_in_split(
    all_codes: list[list[str]], split_codes: list[list[str]]
) -> float:
    """Find percentage of labels that are not in the split. Used to validate the splits"""
    all_codes_unique = {code for codes in all_codes for code in codes}
    split_codes_unique = {code for codes in split_codes for code in codes}
    labels_not_in_split = all_codes_unique - split_codes_unique
    return len(labels_not_in_split) * 100 / len(all_codes_unique)


### Class Definition Starts Here.

class MIMIC3_ICD9:
    def __init__(self, 
                 download_path, 
                 processed_path=None, 
                 lower=True, 
                 remove_special_characters_mullenbach=True, 
                 remove_special_characters=False,
                 remove_digits=True, 
                 remove_accents=False, 
                 remove_brackets=False, 
                 convert_danish_characters=False,
                 min_target_count=10,
                 test_size=0.15,
                 val_size=0.1,
                 step_size=0.2):
        
        self._initialize_attributes(download_path, processed_path, lower, remove_special_characters_mullenbach,
                                    remove_special_characters, remove_digits, remove_accents, remove_brackets,
                                    convert_danish_characters, min_target_count, test_size, val_size, step_size)
        
        if not self._processed_data_exists():
            self._process_and_save_data()
            
        self._load_processed_data()

        
        self.code_system2code_counts = get_code_system2code_counts(self.df, self.code_column_names)     
        
    def _initialize_attributes(self, download_path, processed_path, lower, remove_special_characters_mullenbach,
                               remove_special_characters, remove_digits, remove_accents, remove_brackets,
                               convert_danish_characters, min_target_count, test_size, val_size, step_size):
        # Initialize all class attributes here
        random.seed(10)
        self.download_path = download_path
        self.processed_path = processed_path or Path("files/data/mimiciii_clean")
        self.code_systems = [
            ("ICD9-DIAG", "DIAGNOSES_ICD.csv", "ICD9_CODE", "icd9_diag"),
            ("ICD9-PROC", "PROCEDURES_ICD.csv", "ICD9_CODE", "icd9_proc"),
        ]
        self.preprocessor = TextPreprocessor(
            lower=lower,
            remove_special_characters_mullenbach=remove_special_characters_mullenbach,
            remove_special_characters=remove_special_characters,
            remove_digits=remove_digits,
            remove_accents=remove_accents,
            remove_brackets=remove_brackets,
            convert_danish_characters=convert_danish_characters,
        )
        self.code_column_names = ['icd9_diag', 'icd9_proc']
        self.min_target_count = min_target_count
        self.test_size = test_size
        self.val_size = val_size
        self.step_size = step_size

    def _processed_data_exists(self):
        return os.path.exists(os.path.join(self.processed_path, "mimiciii_clean.feather"))

    def _process_and_save_data(self):
        # Process the data and save it
        print("Processing Clinical Notes and ICD9 Codes!")
        self._create_processed_directory()
        full_dataset = self._prepare_full_dataset()
        self._save_full_dataset(full_dataset)
        self._generate_and_save_splits(full_dataset)

    def _create_processed_directory(self):
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def _prepare_full_dataset(self):
        mimic_notes = self._load_mimic_notes()
        discharge_summaries = prepare_discharge_summaries(mimic_notes)
        merged_codes = download_and_preprocess_code_systems(self.code_systems, self.download_path)
        full_dataset = self._merge_and_filter_dataset(discharge_summaries, merged_codes)
        return full_dataset

    def _load_mimic_notes(self):
        mimic_notes_path = os.path.join(self.download_path, "NOTEEVENTS.csv")
        return pd.read_csv(mimic_notes_path, low_memory=False)

    def _merge_and_filter_dataset(self, discharge_summaries, merged_codes):
        full_dataset = discharge_summaries.merge(
            merged_codes, on=[Config.SUBJECT_ID_COLUMN, Config.ID_COLUMN], how="inner"
        )
        full_dataset = replace_nans_with_empty_lists(full_dataset)
        full_dataset = filter_codes(
            full_dataset, [Config.TARGET_COLUMN, "icd9_proc", "icd9_diag"], min_count=self.min_target_count
        )
        full_dataset = full_dataset[full_dataset[Config.TARGET_COLUMN].apply(len) > 0]
        full_dataset = preprocess_documents(df=full_dataset, preprocessor=self.preprocessor)
        return full_dataset

    def _save_full_dataset(self, full_dataset):
        full_dataset = full_dataset.reset_index(drop=True)
        full_dataset.to_feather(self.processed_path / "mimiciii_clean.feather")
        logging.info(f"Saved full dataset to {self.processed_path / 'mimiciii_clean.feather'}")

    def _generate_and_save_splits(self, full_dataset):
        logging.basicConfig(level=logging.INFO)
        
        # Generate main splits
        splits = self._create_main_splits(full_dataset)
        self._save_splits(splits, "mimiciii_clean_splits.feather")
        # Generate and save subsplits
        self._generate_and_save_subsplits(full_dataset, splits)

    def _create_main_splits(self, full_dataset):
        splits = full_dataset[[Config.SUBJECT_ID_COLUMN, Config.ID_COLUMN]]
        splits['split'] = ''  # Create the 'split' column with empty strings
        subject_series = full_dataset.groupby(Config.SUBJECT_ID_COLUMN)[Config.TARGET_COLUMN].sum()
        subject_ids = subject_series.index.to_list()
        codes = subject_series.to_list()

        subject_ids_train, subject_ids_test = iterative_stratification(
            subject_ids, codes, [1 - self.test_size, self.test_size]
        )
        codes_train = [codes[subject_ids.index(subject_id)] for subject_id in subject_ids_train]
        val_size = self.val_size / (1 - self.test_size)
        subject_ids_train, subject_ids_val = iterative_stratification(
            subject_ids_train, codes_train, [1 - val_size, val_size]
        )

        splits.loc[splits[Config.SUBJECT_ID_COLUMN].isin(subject_ids_train), "split"] = "train"
        splits.loc[splits[Config.SUBJECT_ID_COLUMN].isin(subject_ids_val), "split"] = "val"
        splits.loc[splits[Config.SUBJECT_ID_COLUMN].isin(subject_ids_test), "split"] = "test"
        self._log_split_statistics(codes, subject_ids, subject_ids_train, subject_ids_val, subject_ids_test)

        return splits[[Config.ID_COLUMN, "split"]].reset_index(drop=True)

    def _log_split_statistics(self, codes, subject_ids, subject_ids_train, subject_ids_val, subject_ids_test):
        codes_train = [codes[subject_ids.index(subject_id)] for subject_id in subject_ids_train]
        codes_val = [codes[subject_ids.index(subject_id)] for subject_id in subject_ids_val]
        codes_test = [codes[subject_ids.index(subject_id)] for subject_id in subject_ids_test]

        logging.info("------------- Splits Statistics -------------")
        logging.info(f"Labels missing in the test set: {labels_not_in_split(codes, codes_test)}")
        logging.info(f"Labels missing in the val set: {labels_not_in_split(codes, codes_val)} %")
        logging.info(f"Labels missing in the train set: {labels_not_in_split(codes, codes_train)} %")
        logging.info(f"Test: KL divergence: {kl_divergence(codes, codes_test)}")
        logging.info(f"Val: KL divergence: {kl_divergence(codes, codes_val)}")
        logging.info(f"Train: KL divergence: {kl_divergence(codes, codes_train)}")
        logging.info(f"Test Size: {len(codes_test) / len(codes)}")
        logging.info(f"Val Size: {len(codes_val) / len(codes)}")
        logging.info(f"Train Size: {len(codes_train) / len(codes)}")

    def _save_splits(self, splits, filename):
        splits.to_feather(self.processed_path / filename)
        logging.info(f"Splits saved to {self.processed_path / filename}")

    def _generate_and_save_subsplits(self, full_dataset, splits):
        logging.info("\n------------- Subsplits Statistics -------------")
        full_dataset = full_dataset.merge(splits, on=Config.ID_COLUMN, how='left')
        train_split = full_dataset[full_dataset["split"] == "train"]
        val_split = full_dataset[full_dataset["split"] == "val"]
        test_split = full_dataset[full_dataset["split"] == "test"]

        train_split = train_split.reset_index(drop=True)
        train_indices = train_split.index.to_list()
        train_codes = train_split[Config.TARGET_COLUMN].to_list()

        ratio = self.step_size
        train_indices_remaining = train_indices.copy()
        train_codes_remaining = train_codes.copy()

        while ratio < 1:
            train_indices_remaining, train_indices_split = self._create_subsplit(
                train_indices_remaining, train_codes_remaining, train_indices, ratio
            )
            train_codes_remaining = [train_codes[index] for index in train_indices_remaining]
            
            self._log_subsplit_statistics(train_codes, train_codes_remaining, ratio)
            
            subsplit = self._create_subsplit_dataframe(
                train_split, val_split, test_split, train_indices_split
            )
            self._save_splits(subsplit, f"mimiciii_clean_subsplit_{ratio:.1f}.feather")
            
            ratio += self.step_size

        logging.info("\nMIMIC III splits successfully created!")

    def _create_subsplit(self, train_indices_remaining, train_codes_remaining, train_indices, ratio):
        return iterative_stratification(
            train_indices_remaining,
            train_codes_remaining,
            [
                1 - len(train_indices) * self.step_size / len(train_indices_remaining),
                len(train_indices) * self.step_size / len(train_indices_remaining),
            ],
        )

    def _log_subsplit_statistics(self, train_codes, train_codes_remaining, ratio):
        logging.info(
            f"Labels missing in the subsplit: {labels_not_in_split(train_codes, train_codes_remaining)} %"
        )

    def _create_subsplit_dataframe(self, train_split, val_split, test_split, train_indices_split):
        train_subsplit = train_split.iloc[train_indices_split]
        subsplit = pd.concat([train_subsplit, val_split, test_split])
        return subsplit[[Config.ID_COLUMN, "split"]].reset_index(drop=True)

    def _load_processed_data(self):
        print("Loading The Processed Data")
        dir = self.processed_path
        self.split_filename = 'mimiciii_clean_splits.feather'
        self.data_filename = 'mimiciii_clean.feather'
        
        with vaex.cache.memory_infinite():  # type: ignore
            df = self._load_data_from_feather()
            splits = self._load_splits_from_feather()
            df = df.join(splits, on=Config.ID_COLUMN, how="inner")
            self.df = df
                   
            # self.data = self._create_data_object(df, schema, code_system2code_counts)
        print("Data Loaded Successfully!")


    def _load_data_from_feather(self):
        columns = [
            Config.ID_COLUMN,
            Config.TEXT_COLUMN,
            Config.TARGET_COLUMN,
            "num_words",
            "num_targets",
        ] + self.code_column_names
        
        return vaex.from_arrow_table(
            pa.feather.read_table(
                self.processed_path / self.data_filename,
                columns=columns,
            )
        )

    def _load_splits_from_feather(self):
        return vaex.from_arrow_table(
            pa.feather.read_table(
                self.processed_path / self.split_filename,
            )
        )

    def _create_schema(self):
        return pa.schema([
            pa.field(Config.ID_COLUMN, pa.int64()),
            pa.field(Config.TEXT_COLUMN, pa.large_utf8()),
            pa.field(Config.TARGET_COLUMN, pa.list_(pa.large_string())),
            pa.field("split", pa.large_string()),
            pa.field("num_words", pa.int64()),
            pa.field("num_targets", pa.int64()),
        ])
    
    def all_targets(self) -> set[str]:
        """Get all the targets in the dataset.

        Returns:
            set[str]: Set of all targets.
        """
        all_codes = set()
        for codesystem in self.code_system2code_counts.values():
            all_codes |= set(codesystem.keys())
        return all_codes
    
    def split_size(self, name: str) -> int:
        """Get the size of a split."""
        return len(self.df[self.df["split"] == name])

    def num_split_targets(self, name: str) -> int:
        """Get the number of targets of a split."""
        return len(self.split_targets(name))
    
    def flatten(self,item):
        if isinstance(item, list):
            return [subitem for i in item for subitem in self.flatten(i)]
        return [item]

    def split_targets(self, name: str) -> set[str]:
        """Get the targets of a split."""
        return set(
            self.flatten(self.df[self.df["split"] == name][Config.TARGET_COLUMN].evaluate().to_pylist())
        )
        
    def info(self) -> dict[str, int]:
        """Get information about the dataset.

        Returns:
            dict[str, int]: Dictionary with information about the dataset.
        """
        # print("DEBUG:", self.df.head(10))
        # print("DEBUG:", self.df.columns)
        # print("DEBUG:", len(self.df))
        # print("DEBUG:", len(self.df[self.df.split == "train"]))
        # print("DEBUG:", len(self.df["split"]))
        # print("DEBUG:", self.df["split"] == "train")
        return {
            "num_classes": len(self.all_targets()),
            "num_examples": len(self.df),
            "num_train_tokens": self.df[self.df["split"] == "train"]["num_words"].sum(),
            "average_tokens_per_example": self.df[self.df["split"] == "train"]["num_words"].sum()
            / len(self.df),
            "num_train_examples": self.split_size("train"),
            "num_val_examples": self.split_size("val"),
            "num_test_examples": self.split_size("test"),
            "num_train_classes": self.num_split_targets("train"),
            "num_val_classes": self.num_split_targets("val"),
            "num_test_classes": self.num_split_targets("test"),
            "average_classes_per_example": sum(
                [
                    sum(codesystem.values())
                    for codesystem in self.code_system2code_counts.values()
                ]
            )
            / len(self.df),
        }
    
    
    # return a list of tuples (clinical note, medical codes)
    def get_split_set(self, split): 
        split_df = self.df[self.df["split"] == split]
        rows_text = split_df[Config.TEXT_COLUMN].evaluate()
        rows_target = split_df[Config.TARGET_COLUMN].evaluate()
        return [(str(rows_text[i].as_py()), [str(item) for item in rows_target[i].values]) for i in range(len(split_df))]
    
    def get_training_set(self):
        return self.get_split_set("train")
    
    def get_validation_set(self):
        return self.get_split_set("val")
    
    def get_test_set(self):
        return self.get_split_set("test")
    
    def truncate_text(self, max_length: int) -> None:
        """Truncate text to a maximum length.

        Args:
            max_length (int): Maximum length of text.
        """
        if max_length is None:
            return

        # Convert Vaex column to PyArrow array
        text = pa.array(self.df[Config.TEXT_COLUMN].values)
        
        # Split text into words
        text_split = pc.utf8_split_whitespace(text)
        
        # Truncate to max_length words
        text_split_truncate = pc.list_slice(text_split, 0, max_length)
        
        # Join words back into text
        text_truncate = pc.binary_join(text_split_truncate, " ")
        
        # Convert PyArrow array to numpy array, allowing copy
        text_truncate_np = np.array(text_truncate)
        
        # Create a new Vaex DataFrame
        new_df = vaex.from_arrays(**{
            **{col: self.df[col].values for col in self.df.columns if col != Config.TEXT_COLUMN},
            Config.TEXT_COLUMN: text_truncate_np
        })
        
        # Replace the old DataFrame with the new one
        self.df = new_df
    

