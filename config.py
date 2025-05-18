import os
import yaml
import pandas as pd
from datetime import datetime



# Load parameters from YAML file
with open('config/parameters.yml', 'r', encoding='utf-8') as file:
    parameters = yaml.safe_load(file)

# Load paths from YAML file
with open('config/paths.yml', 'r', encoding='utf-8') as file:
    paths = yaml.safe_load(file)

# Load factors from YAML file
with open('config/factors.yml', 'r', encoding='utf-8') as file:
    factors = yaml.safe_load(file)

class Config:
    """Configuration parameters for the model"""
    # Model parameters
    model_parameters = parameters['model']
    data = parameters['data']
    training_parameters = parameters['training']

    learning_rate_scheduler_parameters = parameters['scheduler']
    augmentation_parameters = parameters['augmentation']
    param_grid = parameters['param_grid']
    random_state = parameters['random_state']
    n_splits = parameters['n_splits']
    patience = parameters['training']['patience']
    sequence_length = parameters['data']['sequence_length']
    



    
    # Date configurations
    TEST_START = parameters['dates']['dates_0']['test_start']
    TEST_END = parameters['dates']['dates_0']['test_end']
    TRAIN_START = parameters['dates']['dates_0']['train_start']
    TRAIN_END = parameters['dates']['dates_0']['train_end']
    
    # Main folder
    MAIN_FOLDER = paths['main_folder']
    
    # Directories
    MODELS_DIR = paths['directories']['model_dir'].format(main_folder=MAIN_FOLDER)
    PLOTS_DIR = paths['directories']['plots_dir'].format(main_folder=MAIN_FOLDER)
    PREDICTIONS_DIR = paths['directories']['predictions_dir'].format(main_folder=MAIN_FOLDER)
    FEATURE_WEIGHTS_DIR = paths['directories']['feature_weights_dir'].format(main_folder=MAIN_FOLDER)
    
    # Files
    SCALER_PATH = paths['files']['scaler_path'].format(main_folder=MAIN_FOLDER)
    MODEL_PATH = paths['files']['model_path'].format(main_folder=MAIN_FOLDER)
    ALL_PRED_PATH = paths['files']['all_pred_path'].format(main_folder=MAIN_FOLDER)
    TEST_PRED_PATH = paths['files']['test_pred_path'].format(main_folder=MAIN_FOLDER)
    ACCURACY_CURVES_PATH = paths['files']['acc_plot_path'].format(main_folder=MAIN_FOLDER)
    BACKTESTING_PATH = paths['files']['backtesting_path'].format(main_folder=MAIN_FOLDER)
    
    # Data paths
    COMBINE_PATH = paths['data']['combine_path']
    RAW_FEATURE_PATH = paths['data']['raw_data_path']
    INDEX_RETURN_PATH = paths['data']['index_return_path']
    CHINESE_VALUATION_DATE_PATH = paths['data']['chinese_valuation_date_path']
    OUTPUT_PATH = paths['data']['output_path']
   
    
    # Backtesting parameters
    BACKTESTING_COST = parameters['backtesting']['cost']
    
    # Feature configurations
    MAJORITY_WINDOW = factors['features']['majority_window']
    TECHNICAL_INDICATORS = factors['features']['technical_indicators']
    DISCRETE_FEATURES = factors['features']['feature_selection']['discrete_features']
    REAL_FEATURES = factors['features']['feature_selection']['real_features']
    
    # Create directories if they don't exist
    @classmethod
    def create_directories(cls):
        directories = [
            cls.MODELS_DIR,
            cls.PLOTS_DIR,
            cls.PREDICTIONS_DIR,
            cls.FEATURE_WEIGHTS_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def get_scaler(cls, X):
        """Get scaler for real features"""
        from sklearn.preprocessing import StandardScaler
        if X.empty or X.shape[1] == 0:  # Handle empty features
            return None
        scaler = StandardScaler()
        scaler.fit(X)
        return scaler

    @classmethod
    def process_dates(cls):
        """Process and split dates into training and testing sets"""
        # Read the Chinese valuation dates
        valuation_dates_path = cls.CHINESE_VALUATION_DATE_PATH
        
        
        try:
            # Try reading Excel first since we know it's an Excel file
            df_dates = pd.read_excel(valuation_dates_path, engine='openpyxl')
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            try:
                # Try CSV as fallback
                df_dates = pd.read_csv(valuation_dates_path)
            except Exception as e:
                print(f"Error reading CSV file: {e}")
                raise
        
        # Convert dates to datetime if needed
        date_column = df_dates.columns[0]  # Assuming the date is in the first column
        df_dates[date_column] = pd.to_datetime(df_dates[date_column])
        
        # Get all dates as a list
        all_dates = df_dates[date_column].tolist()
        
        # Use test period dates from Config class
        test_start = pd.to_datetime(cls.TEST_START)
        test_end = pd.to_datetime(cls.TEST_END)
        train_start = pd.to_datetime(cls.TRAIN_START)
        train_end = pd.to_datetime(cls.TRAIN_END)
        
        # Filter dates to only include dates within the specified range
        valid_dates = [date for date in all_dates if train_start <= date <= test_end]
        
        # Get test period dates
        test_dates = [date for date in valid_dates if test_start <= date <= test_end]
        
        # Get training dates (using specified training period)
        train_dates = [date for date in valid_dates if train_start <= date <= train_end]
        
        # Sort the dates
        train_dates.sort()
        test_dates.sort()
        
      
        
        return {
            'all_dates': valid_dates,
            'test_dates': test_dates,
            'train_dates': train_dates
        } 

    @classmethod
    def load_date_config(cls, config_name='dates'):
        """Load date configuration from parameters.yml"""
        with open('config/parameters.yml', 'r', encoding='utf-8') as file:
            params = yaml.safe_load(file)
        return params['model'][config_name]

    @classmethod
    def update_config_dates(cls, date_config):
        """Update Config class with the selected date configuration"""
        cls.TEST_START = date_config['test_start']
        cls.TEST_END = date_config['test_end']
        cls.TRAIN_START = date_config['train_start']
        cls.TRAIN_END = date_config['train_end'] 