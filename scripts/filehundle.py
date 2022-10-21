from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
import pandas as pd

from scripts.logger import get_logger
my_logger = get_logger("FileHandler")
my_logger.debug("Loaded successfully!")

class FileHandler():

  def __init__(self):
    pass
 
  def save_csv(self, df, csv_path, index=False):
    try:
      df.to_csv(csv_path, index=index)
      my_logger.info("file saved as csv")

    except Exception:
      my_logger.exception("save failed")


  def read_csv(self, csv_path):
    try:
      df = pd.read_csv(csv_path)
      my_logger.debug("file read as csv")
      return df
    except FileNotFoundError:
      my_logger.exception("file not found")


  def normalizer(self, df, columns):
        norm = Normalizer()
        return pd.DataFrame(norm.fit_transform(df), columns=columns)


  def scaler(self, df, columns, mode="minmax"):
    if (mode == "minmax"):
        minmax_scaler = MinMaxScaler()
        return pd.DataFrame(minmax_scaler.fit_transform(df), columns=columns)

    elif (mode == "standard"):
      scaler = StandardScaler()
      return pd.DataFrame(scaler.fit_transform(df), columns=columns)


  def scale_and_normalize(self, df, scaler="minmax"):
    columns = df.columns
    return self.normalizer(self.scaler(df, columns, scaler), columns)