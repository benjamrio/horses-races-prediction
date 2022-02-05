import pandas as pd
from tqdm import tqdm
import numpy as np
import re
import glob
import json
import utils.flatten as flt
import argparse


def glob_to_df(files_paths_glob, selected_columns):
    selected_columns = [
        "raceId",
        "date",
        "horse.genyId",
        "musique",
        "results.position",
        "priceFirst",
    ]
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
    return pd.DataFrame(performances)[selected_columns]


def preprocess_df(df):
    print(
        f"Preprocessing the horse performances dataframe containing {len(df)} rows..."
    )
    df["date"] = pd.to_datetime(df["date"], unit="ms")
    df["results.position"] = df["results.position"].apply(clean_results_position)
    df["cleaned_music"] = df["musique"].apply(clean_music_to_list)
    return df


def clean_results_position(results_position):
    if pd.isnull(results_position):
        return 10
    else:
        return max(int(results_position), 10)


def clean_music_to_list(musique):
    """Nettoie la musique (retire les lettres et supprime les nombres entre parenthèses)"""
    if pd.isnull(musique):
        return []

    musique = re.sub(r"\([^)]*\)", "", musique)
    musique = re.sub("[^1-9]", " ", musique)
    musique = musique.split()
    return musique


def get_music(horse):
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
    print("Retrieving augmented music for each horse...")
    return df.groupby(["horse.genyId"]).apply(get_music)


def parse_args():
    my_parser = argparse.ArgumentParser(description="Builds augmented music of horses")
    my_parser.add_argument(
        "--input_glob",
        "-i",
        default="data/raw/2016-2018_races/historic/*",
        help="Input JSON files glob pattern",
    )
    my_parser.add_argument(
        "--output_folder",
        "-o",
        default="data/interim/augmented_musics.csv",
        help="Path of the output directory, in which the result file augmented_musics.csv will be created",
    )
    args = parse_args()


def main():

    args = parse_args()
    get_augmented_music_df(
        preprocess_df(glob_to_df(args.input_glob, selected_columns))
    ).to_csv(args.output_folder)


if __name__ == "__main__":
    main()
