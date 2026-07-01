import pandas as pd
import json
from pathlib import Path

RAW_PATH = "data/raw/employee_np.xlsx"
OUTPUT_PATH = "data/processed/employee_chunks.json"


def load_and_clean_data():
    # read the excel file into a dataframe
    df = pd.read_excel(RAW_PATH)

    # some text fields had extra spaces, so clean those up
    text_columns = df.select_dtypes(include="object").columns
    for col in text_columns:
        df[col] = df[col].str.strip()

    # dates were stored as full timestamps, we just want something readable like "March 2023"
    df["Hire_Date"] = pd.to_datetime(df["Hire_Date"]).dt.strftime("%B %Y")

    return df


def row_to_sentence(row):
    # this is the important part - turning a row of numbers/labels into a sentence
    # so we can later search it semantically instead of just filtering columns
    sentence = (
        f"{row['Full_Name']} (ID {row['Employee_ID']}) works as a {row['Job_Title']} "
        f"in the {row['Department']} department. Hired {row['Hire_Date']}, based in "
        f"{row['Location']}. Current status: {row['Status']}, working {row['Work_Mode']}. "
        f"{row['Experience_Years']} years of experience, performance rating "
        f"{row['Performance_Rating']}/5, salary {row['Salary_INR']:,} INR."
    )
    return sentence


def build_chunks(df):
    # one employee = one chunk. rows are short enough that splitting them further
    # would just break the info apart (e.g. separating name from salary), so we don't
    chunks = []
    for _, row in df.iterrows():
        chunks.append({
            "id": row["Employee_ID"],
            "text": row_to_sentence(row)
        })
    return chunks


def main():
    df = load_and_clean_data()
    chunks = build_chunks(df)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Processed {len(chunks)} employees, saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()