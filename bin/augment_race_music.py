import pandas as pd
from tqdm import tqdm
import numpy as np
import re
import glob
import json
import utils.flatten as flt
import argparse
from pathlib import Path
import os
import math

"""
Augment JSON input files with augmented music at the (chronological) end of the input files.
"""


def glob_to_df(files_paths_glob):
    """Aggregate JSON files containing horse race data described by the glob pattern into a DataFrame, with desired columns

    Arguments:
        files_paths_glob {string} -- Glob file pattern of historic JSON files. Should be complete (end with .json)

    Returns:
        DataFrame -- Historic performances of horses
    """
    if files_paths_glob == None:
        return pd.DataFrame()
    files_paths = glob.glob(files_paths_glob)
    print(f"Data extraction from {len(files_paths)} JSON files...")
    pbar = tqdm(
        total=len(files_paths),
        desc="Computing augmented music of each horse based on archive files ...",
    )
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
    # try:
    # df["date"] = pd.to_datetime(df["date"], unit="ms")   Otherwise not seriazable
    # except KeyError:
    # print("Cannot find date in augmented music df")
    try:
        df["results.position"] = df["results.position"].apply(
            clean_results_position)
    except KeyError:
        print("Cannot find result position in augmented music df")
    try:
        df["cleaned_music"] = df["musique"].apply(clean_music_to_list)
    except KeyError:
        print("Cannot find musique  in augmented music df")
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
       Series -- Output of the program : id, augmented music
    """
    print("Retrieving augmented music for each horse...")
    try:
        series = df.groupby(["horse.genyId"]).apply(get_music)
        series = series.reindex(series.index.values.astype(int))
    except KeyError:
        print("Cannot find horse.genyId, initializing empty series")
        series = pd.Series(dtype=object)
    return series


def parse_args():
    """Create the argument parser to retrieve input and output

    Returns:
        Namespace -- arguments
    """
    my_parser = argparse.ArgumentParser(
        description="Builds augmented music of horses")
    my_parser.add_argument(
        "--input",
        "-i",
        # default="data/raw/2016-2018_races/historic/*.json",
        required=True,
        help="Input JSON files glob pattern. Files to which the computed and actualized augmented music will be added to",
    )
    my_parser.add_argument(
        "--output",
        "-o",
        # default="data/interim/augmented_musics.csv",
        required=True,
        help="Path of the output directory, in which the modified input files will be saved to",
    )
    my_parser.add_argument(
        "--archive",
        "-a",
        help="Path of the files containing historic data to augment the music. Archive files must be older than inputs.",
    )
    args = my_parser.parse_args()
    return args


def update_inputs(historic, input, output):
    """Updates the input JSON files based on the augmented music computed from the archive files,
    all the while permanentyl updating augmented music database.

    Arguments:
        historic_df {Series} -- Contains the augmented music of each horse, based on the data of the archive files
        input {string} -- glob string of the files to which the augmented music will be added to. Given by the arg parser
        output {string} -- glob string of the output directory to which the updated input files will be added to. Given by the arg parser
    """
    input_paths = glob.glob(input)
    print(f"Augmenting {len(input_paths)} files ...")
    pbar = tqdm(
        total=len(input_paths),
        desc="Updating input files, saving them to output directory ...",
    )
    counter_no_id = 0
    # iteration on every input file
    for input_path in input_paths:
        outfile_path = os.path.join(output, Path(input_path).name)

        # read input file, write output file
        with open(input_path, "r", encoding="utf-8") as input_file, open(
            outfile_path, "w+", encoding="utf-8"
        ) as output_file:
            # copy initial data
            data = json.load(input_file)
            pbar.update()
            # get race info for updating augmented music db
            try:
                priceFirst = data["price"]["first"]
            except:
                priceFirst = np.NaN
            date = data["raceScheduledStartEpochMs"]

            partants = []
            for horse in data["partants"]:
                try:
                    id = horse["horse"]["genyId"]
                except:
                    counter_no_id += 1
                    partants.append(horse)
                    continue  # If a horse has no id we switch to the next one

                dai = False
                # update augmented music databases with this race's result
                try:
                    result = horse["results"]["position"]
                except:
                    dai = True
                    result = None

                performance = {"position": result,
                               "priceFirst": priceFirst, "date": date, "dai": dai}
                # Si la musique augmentée existe déjà dans la bdd historic
                # on récupère les perforamnces passées
                # et on ajoute la performance d'aujourd'hui
                augmented_music = []
                if id in historic:

                    for histo in historic[id]:
                        position = int(histo[0])
                        item = {"position": position}
                        if math.isnan(position):
                            raise Exception('no position')

                        if not math.isnan(histo[1]):
                            item['priceFirst'] = histo[1]

                        if not math.isnan(histo[2]):
                            item['date'] = histo[2]

                        augmented_music.append(item)
                        historic[id] = historic[id].append(performance)

                else:
                    historic[id] = [performance]
                horse["augmentedMusic"] = list(augmented_music)
                partants.append(horse)
            data["partants"] = partants
            json.dump(data, output_file)
    pbar.close()
    print(f"\n{counter_no_id} horses had no genyId, they were skipped in the process")


def write_outfile(input_path, augmented_musics_dic, output_path):
    outfile_path = 0
    with open(input_path, "r", encoding="utf-8") as input_file:
        json.dump()


def main():
    args = parse_args()
    historic_df = get_augmented_music_df(
        preprocess_df(glob_to_df(args.archive)))
    update_inputs(historic_df, args.input, args.output)


if __name__ == "__main__":
    main()
