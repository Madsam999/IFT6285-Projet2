import pandas as pd
import os

DIRECTORY_OF_RESULTS = "results"
DIRECTORY_OF_CSV = "csv"
PATH_TO_RELATIONS = os.path.join(DIRECTORY_OF_RESULTS, DIRECTORY_OF_CSV, "relations.csv")

def clean_relations():
    relations_df = pd.read_csv(PATH_TO_RELATIONS, sep = ";")

    totalRows = len(relations_df)
    duplicateRows = relations_df.duplicated().sum()
    percentage = (duplicateRows / totalRows) * 100

    print(f"Total Rows: {totalRows}")
    print(f"Duplicate Count: {duplicateRows}")
    print(f"Duplicate Percentage: {percentage}")

    relations_df.drop_duplicates(inplace = True)

    relations_df.to_csv("results/csv/relations_clean.csv", sep = ";", index = False)

    return relations_df


if __name__ == "__main__":
    clean_relations()