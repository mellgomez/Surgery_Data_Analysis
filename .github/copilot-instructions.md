# AI Agent Instructions for CW_460 Project

## Project Overview
This is a data analysis project focused on processing surgery timing data from Excel files using Python and pandas. The main functionality involves reading and analyzing data from `SurgeryTiming.xlsx`.

## Key Components

### Data Processing Script (`460_cw.py`)
- Main script for data analysis
- Uses pandas for Excel file operations
- Current capabilities:
  - Reading Excel files
  - Displaying basic data info (headers and data types)
  - Support for multiple sheet handling

## Development Environment

### Dependencies
- Python
- pandas library
- Required Excel file: `SurgeryTiming.xlsx` in the same directory as the script

### Project Structure
```
CW_460/
├── 460_cw.py              # Main analysis script
└── SurgeryTiming.xlsx     # Data source file
```

## Coding Patterns and Conventions

### Data Loading
- Use pandas' `read_excel()` for reading Excel files
- Example:
  ```python
  df = pd.read_excel(file_path)
  ```
- When working with specific sheets, use the `sheet_name` parameter:
  ```python
  df = pd.read_excel(file_path, sheet_name='SheetName')
  ```

### Data Exploration
- Use `df.head()` for initial data preview
- Use `df.dtypes` to check column data types

## Key Files
- `460_cw.py`: Main script containing data analysis logic
- `SurgeryTiming.xlsx`: Source data file (required to be in the same directory)