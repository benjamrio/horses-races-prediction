import pandas as pd
from tqdm import tqdm
import numpy as np
import re
import glob
import json
import utils.flatten as flt
import argparse


def glob_to_df(files_paths_glob):
    """Aggregate JSON files containing horse race data described by the glob pattern into a DataFrame, with desired columns

    Arguments:
        files_paths_glob {string} -- Glob file pattern of historic JSON files. Should be complete (end with .json)

    Returns:
        DataFrame -- Historic performances of horses
    """
    files_paths = glob.glob(files_paths_glob)
    print(f"Data extraction from {len(files_paths)} JSON files...")
    pbar = tqdm(total=len(files_paths))
    performances = []
    for file_path in files_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            pbar.update()
            for horse in data["partants"]:
                horse_performance = flt.flatten_dic(horse)
                horse_performance["date"] = data["raceScheduledStartEpochMs"]
                try:
                    horse_performance["priceFirst"] = data["price"]["first"]
                except:
                    pass

                # Des fichiers n'ont pas le genyId comme raceId, mais un nom courant
                try:
                    horse_performance["raceId"] = data["genyId"]
                except:
                    pass

                performances += [horse_performance]
    pbar.refresh()
    selected_columns = [
        "raceId",
        "date",
        "horse.genyId",
        "musique",
        "results.position",
        "priceFirst",
    ]
    return pd.DataFrame(performances)[selected_columns]


def preprocess_df(df):
    """Preprocess an historic performances DataFrame.
        Modifying :
            - date : to datetime
            - result : ceil the position to 10, attribute 10 to a non ranked performance
            - music : keep the result positions only, to a list

    Arguments:
        df {DataFrame} -- Historic horse performances

    Returns:
        df -- Preprocessed DataFrame
    """
    print(
        f"Preprocessing the horse performances dataframe containing {len(df)} rows..."
    )
    df["date"] = pd.to_datetime(df["date"], unit="ms")
    df["results.position"] = df["results.position"].apply(clean_results_position)
    df["cleaned_music"] = df["musique"].apply(clean_music_to_list)
    return df


def clean_results_position(result_position):
    """Ceil the result position to 10 and attribute 10 to a non ranked performance

    Arguments:
        result_position {string} -- The arriving rank of the horse (can be 'D' for disqualified)

    Returns:
        int -- corrected result position
    """
    if pd.isnull(result_position):
        return 10
    else:
        return max(int(result_position), 10)


def clean_music_to_list(music):
    """Supress letters (for race type) and new year indication (number between parentheses)

    Arguments:
        music {string} -- Music as given by websites
        (a word alterning numbers and letters)

    Returns:
        int list -- cleaned music
    """
    if pd.isnull(music):
        return []

    musique = re.sub(r"\([^)]*\)", "", music)
    musique = re.sub("[^1-9]", " ", music)
    musique = musique.split()
    return musique


def get_music(horse):
    """Creates augmented music from one horse's performances

    Arguments:
        horse {DataFrame} -- Horses performances, in chronological order

    Returns:
        2D np.array -- augmented music with date, cash prize, and position.
        Each element of the list is a performance, which is a list of the 3 features
    """
    first_music = horse["cleaned_music"].iloc[0]
    augmented_music = horse[["results.position", "priceFirst", "date"]].values
    if first_music:
        augmented_music = np.vstack(
            [
                [[float(result), np.NaN, np.NaN] for result in first_music],
                augmented_music,
            ]
        )
    return augmented_music


def get_augmented_music_df(df):
    """Retrieve the augmented music for each horse appearing in the horses' performances dataframe

    Arguments:
        df {DataFrame} -- Horses performances

    Returns:
        DataFrame -- Output of the program : DataFrame with
    """
    print("Retrieving augmented music for each horse...")
    return df.groupby(["horse.genyId"]).apply(get_music)


def parse_args():
    """Create the argument parser to retrieve input and output

    Returns:
        Namespace -- arguments
    """
    my_parser = argparse.ArgumentParser(description="Builds augmented music of horses")
    my_parser.add_argument(
        "--input",
        "-i",
        default="data/raw/2016-2018_races/historic/*.json",
        help="Input JSON files glob pattern",
    )
    my_parser.add_argument(
        "--output",
        "-o",
        default="data/interim/augmented_musics.csv",
        help="Path of the output directory, in which the result file augmented_musics.csv will be created",
    )
    args = my_parser.parse_args()
    return args


def main():
    args = parse_args()
    get_augmented_music_df(preprocess_df(glob_to_df(args.input))).to_csv(args.output)


if __name__ == "__main__":
    main()
