import pandas as pd


def read_csv_as_chunk(fname, sample_size, chunk_size=1000):
    reader = pd.read_csv(fname, header=0, nrows=sample_size,
                         iterator=True, low_memory=False)
    chunks = []
    loop = True
    while loop:
        try:
            chunk = reader.get_chunk(chunk_size)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Finish reading csv. Iteration is stopped")

    df_ac = pd.concat(chunks, ignore_index=True)
    return df_ac


def save_df_to_csv(input_df: pd.DataFrame(), to_filename: str, to_path=DATA_SOURCE) -> None:
    file_with_path = f"{to_path}/{to_filename}"
    try:
        input_df.to_csv(f"{file_with_path}", index=False)
        print(f"U have successfully save file {file_with_path}")
        aa
    except Exception as e:
        print("Fail to save csv file")
        raise e
