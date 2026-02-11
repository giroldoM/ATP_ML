import pandas as pd
import numpy as np
from pathlib import Path
from dataio import DataIOConfig, read_range

# --- CONFIGURATION ---
# 1. Get the location of this file (features.py) -> .../ATP_ML/ML
# 2. Go up two levels (.parent.parent) -> .../ATP_ML
# 3. Enter the data folder -> .../ATP_ML/data
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# We keep drop_leakage=False for now to access 'score' if needed for feature engineering,
# but we must be careful to drop it before training.
cfg = DataIOConfig(data_dir=DATA_DIR, drop_leakage=False)

def create_pairwise_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the dataset from (Winner, Loser) format to (Player1, Player2) format.
    
    This doubles the dataset size:
    - Row 1: P1 = Winner, P2 = Loser, Target = 1
    - Row 2: P1 = Loser, P2 = Winner, Target = 0
    
    Returns:
        pd.DataFrame: A dataframe ready for training with 'target' column.
    """
    print("🔄 Reformatting data to Player1 vs Player2 structure...")
    
    # 1. Create the "Positive" samples (Player 1 is the Winner)
    # We rename columns starting with 'winner_' to 'p1_' and 'loser_' to 'p2_'
    df_pos = df.copy()
    cols_pos = {c: c.replace('winner_', 'p1_').replace('loser_', 'p2_') 
                for c in df_pos.columns if 'winner_' in c or 'loser_' in c}
    df_pos = df_pos.rename(columns=cols_pos)
    df_pos['target'] = 1  # Player 1 won

    # 2. Create the "Negative" samples (Player 1 is the Loser)
    # We rename columns starting with 'loser_' to 'p1_' and 'winner_' to 'p2_'
    df_neg = df.copy()
    cols_neg = {c: c.replace('loser_', 'p1_').replace('winner_', 'p2_') 
                for c in df_neg.columns if 'winner_' in c or 'loser_' in c}
    df_neg = df_neg.rename(columns=cols_neg)
    df_neg['target'] = 0  # Player 1 lost (Player 2 won)

    # 3. Combine both dataframes
    df_pairwise = pd.concat([df_pos, df_neg], axis=0, ignore_index=True)
    
    # 4. Sort by date to maintain temporal integrity (crucial for time-series splits)
    if 'date' in df_pairwise.columns:
        df_pairwise = df_pairwise.sort_values(by=['date', 'match_num']).reset_index(drop=True)
        
    return df_pairwise

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds predictive features (Math & Statistics) to the dataset.
    """
    print("🛠️  Engineering features (Ranks, Age, Surface, Round)...")
    df = df.copy()

    # --- 1. Handle Missing Values (Null Ranks) ---
    # If a player has no rank, we give them a very low rank (e.g., 9999)
    df['p1_rank'] = df['p1_rank'].fillna(9999)
    df['p2_rank'] = df['p2_rank'].fillna(9999)
    
    # Fill missing ages with the mean age of all players
    df['p1_age'] = df['p1_age'].fillna(df['p1_age'].mean())
    df['p2_age'] = df['p2_age'].fillna(df['p2_age'].mean())

    # --- 2. Create Diff Features ---
    # Rank Diff: Positive means P1 has a BETTER (lower) rank than P2
    # Logic: If P1 is Rank 1 and P2 is Rank 100 -> P2(100) - P1(1) = 99 (Positive = P1 Favorite)
    df['rank_diff'] = df['p2_rank'] - df['p1_rank']
    
    # Age Diff: P1 age - P2 age
    df['age_diff'] = df['p1_age'] - df['p2_age']

    # --- 3. Encode Categorical Data (Text -> Numbers) ---
    # Surface: Hard=0, Clay=1, Grass=2, Carpet=3
    surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
    df['surface_code'] = df['surface'].map(surface_map).fillna(-1)

    # Hand: R=0, L=1, U=2 (Unknown)
    df['p1_hand_code'] = df['p1_hand'].map({'R': 0, 'L': 1, 'U': 2}).fillna(0)
    df['p2_hand_code'] = df['p2_hand'].map({'R': 0, 'L': 1, 'U': 2}).fillna(0)

    # --- 4. Round Encoding (Hierarchy of Importance) ---
    # We assign higher values to later rounds
    round_map = {
        'F': 7,      # Final (Most important)
        'SF': 6,     # Semi-Final
        'QF': 5,     # Quarter-Finals
        'R16': 4,    # Round of 16
        'R32': 3,    # Round of 32
        'R64': 2,    # Round of 64
        'R128': 1,   # Round of 128 (First round)
        'RR': 0,     # Round Robin / Others
        'BR': 6      # Bronze Medal (Olympics) - equivalent to Semi
    }
    # Map text to numbers. fillna(0) handles rare cases.
    df['round_code'] = df['round'].map(round_map).fillna(0)

    return df

def prepare_for_model(df: pd.DataFrame):
    """
    Selects only the columns that the model can see.
    Removes names, dates, scores, and IDs.
    """
    # List of features we created + some basics
    features = [
        'rank_diff', 'age_diff', 
        'p1_rank', 'p2_rank', 
        'p1_age', 'p2_age',
        'surface_code', 'p1_hand_code', 'p2_hand_code',
        'round_code' # Using the encoded round
    ]
    
    # Check which features actually exist in df (safety check)
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features]
    y = df['target']
    
    return X, y

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"📂 Loading data from: {DATA_DIR}")

    try:
        # 1. Load Data (Using a larger range for a proper test: 2015-2023)
        df_raw = read_range(cfg, 2015, 2023)
        print(f"✅ Raw Data Loaded: {df_raw.shape[0]} matches.")

        # 2. Structure (Winner/Loser -> Player1/Player2)
        df_pairwise = create_pairwise_data(df_raw)
        
        # 3. Add Features (Math & Encodings)
        # We pass df_pairwise to the function for clarity
        df_rich = add_features(df_pairwise)
        
        # 4. Cleanup for Model
        X, y = prepare_for_model(df_rich)
        
        print("\n✅ READY FOR TRAINING!")
        print(f"Features Matrix (X): {X.shape}")
        print(f"Target Vector (y):   {y.shape}")
        
        print("\n--- Example of Model Input (First 5 rows) ---")
        print(X.head())

    except Exception as e:
        print(f"❌ ERROR: {e}")