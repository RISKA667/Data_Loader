import os
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm
from datetime import datetime
import time

class OHLCVFormatter:
    def __init__(self, input_dir: str, output_dir: str = None):
        """
        Initialise le formateur OHLCV.
        
        Args:
            input_dir (str): Chemin du dossier contenant les fichiers à traiter
            output_dir (str, optional): Chemin du dossier de sortie
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir.parent / 'RESULTAT'
        self.output_dir.mkdir(exist_ok=True)
        
    def validate_timestamp(self, timestamp: any) -> Optional[int]:
        """
        Valide et corrige le format du timestamp.
        Convertit en millisecondes pour compatibilité avec backtrader.
        
        Args:
            timestamp (any): Timestamp à valider (peut être un nombre ou une chaîne de date)
            
        Returns:
            Optional[int]: Timestamp en millisecondes ou None si invalide
        """
        try:
            # Si c'est une chaîne de caractères qui ressemble à une date
            if isinstance(timestamp, str) and not timestamp.replace('.', '').isdigit():
                try:
                    # Parser la date et convertir en millisecondes
                    dt = pd.to_datetime(timestamp)
                    ts = int(dt.timestamp() * 1000)
                except:
                    return None
            else:
                # Conversion en float pour gérer les formats numériques
                ts = float(timestamp)
                
                # Vérification que le timestamp n'est pas négatif
                if ts <= 0:
                    return None
                
                # Obtenir la longueur originale avant conversion
                original_length = len(str(int(ts)))
                
                # Conversion et validation selon le format
                if original_length > 16:  # Timestamp trop long
                    return None
                elif original_length >= 15:  # Probablement en microsecondes
                    ts = ts / 1000  # Conversion en millisecondes
                elif original_length <= 10:  # En secondes
                    ts = ts * 1000  # Conversion en millisecondes
            
            # Après conversion, vérifier si le timestamp est dans une plage valide
            # Entre 2010-01-01 et 2050-01-01 (en millisecondes)
            min_ts = 1262304000000  # 2010-01-01 00:00:00
            max_ts = 2524608000000  # 2050-01-01 00:00:00
            
            if min_ts <= ts <= max_ts:
                return int(ts)
            
            return None
            
        except (ValueError, TypeError):
            return None

    def validate_ohlcv(self, row) -> bool:
        """
        Valide les données OHLCV d'une ligne.
        
        Args:
            row: Ligne de données OHLCV [timestamp, open, high, low, close, volume]
            
        Returns:
            bool: True si la ligne est valide, False sinon
        """
        try:
            # Vérification du timestamp
            timestamp = self.validate_timestamp(row[0])
            if timestamp is None:
                return False
            
            # Extraire les valeurs OHLCV
            try:
                _, open_price, high, low, close, volume = [float(x) for x in row]
            except (ValueError, TypeError):
                return False
            
            # Validation des règles OHLCV
            if any(pd.isna(x) for x in [open_price, high, low, close, volume]):
                return False
            if high < low:
                return False
            if open_price < low or open_price > high:
                return False
            if close < low or close > high:
                return False
            if volume < 0:
                return False
                
            return True
        except:
            return False

    def format_ohlcv_row(self, row) -> Optional[str]:
        """
        Formate une ligne en format OHLCV standardisé.
        
        Args:
            row: Ligne de données à formater
            
        Returns:
            Optional[str]: Ligne formatée ou None si erreur
        """
        try:
            # Validation et correction du timestamp (en millisecondes)
            timestamp = self.validate_timestamp(row[0])
            if timestamp is None:
                return None
            
            # Conversion et arrondi des valeurs OHLCV
            try:
                values = [float(x) for x in row[1:6]]
                formatted_values = [f"{value:.4f}" for value in values]
                return f"{timestamp},{','.join(formatted_values)}"
            except (ValueError, TypeError, IndexError):
                return None
        except:
            return None

    def extract_zip_files(self) -> List[Path]:
        """
        Extrait tous les fichiers ZIP du dossier d'entrée.
        
        Returns:
            List[Path]: Liste des chemins des fichiers extraits
        """
        extracted_files = []
        zip_files = list(self.input_dir.glob('*.zip'))
        
        if not zip_files:
            return []

        print("\nExtracting ZIP files...")
        for zip_path in tqdm(zip_files, desc="ZIP files"):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zipf:
                    csv_files = [f for f in zipf.namelist() if f.lower().endswith('.csv')]
                    for csv_file in csv_files:
                        try:
                            zipf.extract(csv_file, self.input_dir)
                            extracted_files.append(Path(self.input_dir / csv_file))
                            print(f"Extracted: {csv_file}")
                        except Exception as e:
                            print(f"Error extracting {csv_file}: {str(e)}")
                            continue
            except Exception as e:
                print(f"Error with ZIP file {zip_path}: {str(e)}")
                continue

        return extracted_files

    def process_csv(self, file_path: Path) -> List[str]:
        """
        Traite un fichier CSV et retourne les lignes formatées.
        
        Args:
            file_path (Path): Chemin du fichier CSV à traiter
            
        Returns:
            List[str]: Liste des lignes valides et formatées
        """
        try:
            print(f"\nProcessing: {file_path}")
            # Lecture du CSV avec pandas
            df = pd.read_csv(file_path, header=None)
            
            if df.empty:
                print(f"File is empty: {file_path}")
                return []
            
            # S'assurer qu'il y a au moins 6 colonnes
            if df.shape[1] < 6:
                print(f"Insufficient columns ({df.shape[1]}): {file_path}")
                return []
                
            # Ne garder que les 6 premières colonnes
            df = df.iloc[:, :6]
            
            # Validation et formatage des lignes
            valid_rows = []
            total_rows = len(df)
            valid_count = 0
            invalid_timestamps = 0
            invalid_data = 0
            
            for _, row in df.iterrows():
                if self.validate_ohlcv(row):
                    formatted_row = self.format_ohlcv_row(row)
                    if formatted_row:
                        valid_rows.append(formatted_row)
                        valid_count += 1
                elif self.validate_timestamp(row[0]) is None:
                    invalid_timestamps += 1
                else:
                    invalid_data += 1
            
            print(f"Processed {total_rows} rows:")
            print(f"  - Valid: {valid_count}")
            print(f"  - Invalid timestamps: {invalid_timestamps}")
            print(f"  - Invalid OHLCV data: {invalid_data}")
            
            return valid_rows
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []

    def compile_files(self):
        """
        Compile tous les fichiers en format OHLCV standardisé.
        Le format de sortie est compatible avec backtrader (timestamps en millisecondes).
        """
        # Extraction des ZIP d'abord
        self.extract_zip_files()
        
        # Recherche de tous les CSV
        csv_files = list(self.input_dir.glob('*.csv'))
        
        if not csv_files:
            print("No CSV files found!")
            return
        
        print(f"\nFound {len(csv_files)} CSV files to process")
        
        # Traitement des CSV
        all_rows = []
        print("\nProcessing CSV files...")
        for csv_file in tqdm(csv_files, desc="Processing"):
            rows = self.process_csv(csv_file)
            all_rows.extend(rows)

        if not all_rows:
            print("No valid data found!")
            return

        # Suppression des doublons et tri par timestamp
        print("\nRemoving duplicates and sorting...")
        original_count = len(all_rows)
        all_rows = list(set(all_rows))  # Suppression des doublons
        duplicate_count = original_count - len(all_rows)
        
        # Tri par timestamp (en millisecondes)
        all_rows.sort(key=lambda x: int(x.split(',')[0]))
        
        # Sauvegarde des résultats
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_csv = self.output_dir / f"combined_ohlcv_{timestamp}.csv"
        output_zip = output_csv.with_suffix('.zip')
        
        print("\nSaving results...")
        # Sauvegarde CSV
        with open(output_csv, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_rows))
        
        # Création du ZIP
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(output_csv, output_csv.name)
        
        print(f"\nProcessing complete!")
        print(f"Total rows processed: {original_count}")
        print(f"Duplicates removed: {duplicate_count}")
        print(f"Final rows: {len(all_rows)}")
        print(f"Output CSV: {output_csv}")
        print(f"Output ZIP: {output_zip}")
        print("\nFormat de sortie compatible avec backtrader:")
        print("  - Timestamps en millisecondes")
        print("  - Format: timestamp,open,high,low,close,volume")
        print("  - Utiliser avec: dtformat=lambda x: datetime.fromtimestamp(int(x) / 1000)")

def main():
    # Modifier ce chemin selon votre configuration
    input_dir = r"C:\Users\Louis\Documents\Développement\DATA_LOAD"
    
    try:
        formatter = OHLCVFormatter(input_dir)
        formatter.compile_files()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nProcess finished")

if __name__ == "__main__":
    main()