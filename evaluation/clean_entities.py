import pandas as pd
import os

DIRECTORY_OF_RESULTS = "results"
DIRECTORY_OF_CSV = "csv"
PATH_TO_RELATIONS = os.path.join(DIRECTORY_OF_RESULTS, DIRECTORY_OF_CSV, "entities.csv")

def clean_entities():
    entities_df = pd.read_csv(PATH_TO_RELATIONS, sep = ";")

    unk_mask = entities_df["entity"].str.lower().isin(["unk", "unkown", "none"])

    total_unk_count = entities_df.loc[unk_mask, "frequency"].sum()
    total_entity_count = entities_df["frequency"].sum()

    if total_entity_count > 0:
        percentage = (total_unk_count / total_entity_count) * 100
    else:
        percentage = 0
    
    print(f"Total Entities Found: {total_entity_count}")
    print(f"Unknown Entities Found: {total_unk_count}")
    print(f"Noise Percentage: {percentage:.2f}%")

    df_cleaned = entities_df[~unk_mask]

    df_cleaned.to_csv(os.path.join(DIRECTORY_OF_RESULTS, DIRECTORY_OF_CSV, "entities_clean.csv"))

    return df_cleaned

if __name__ == "__main__":
    clean_entities()