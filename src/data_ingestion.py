from pathlib import Path
import pandas as pd


def load_raw_data(file_path: str) -> pd.DataFrame:
    '''
    Load raw data from a CSV file.
    '''
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')

    df = pd.read_csv(path)

    print(f'Data loaded successfully from: {path}')

    return df


def validate_schema(df: pd.DataFrame) -> bool:
    '''
    Basic schema validation.
    '''
    if df.empty:
        raise ValueError('The dataset is empty.')

    print('Schema validation: PASSED')

    return True


def save_processed_data(df: pd.DataFrame, file_path: str) -> None:
    '''
    Save the DataFrame to a CSV file.
    '''
    path = Path(file_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)

    print(f'Processed data saved to: {path}')


def main():
    raw_path = r'C:\Users\anush\OneDrive\Desktop\churnIQ\data\raw\Telco.csv'

    processed_path = (
        r'C:\Users\anush\OneDrive\Desktop\churnIQ\data\processed\churn_processed.csv'
    )

    # Load data
    df = load_raw_data(raw_path)

    # Validate
    validate_schema(df)

    # Print dataset information
    print('\nDataset Information')
    print('-' * 30)
    print(f'Rows: {df.shape[0]}')
    print(f'Columns: {df.shape[1]}')

    # Print column names
    print('\nColumn Names')
    print('-' * 30)
    print(df.columns.tolist())

    # Print missing values
    print('\nMissing Values')
    print('-' * 30)
    print(df.isnull().sum())

    # Save processed data
    save_processed_data(df, processed_path)


if __name__ == '__main__':
    main()