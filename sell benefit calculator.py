import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
import tkinter as tk
from tkinter import filedialog
import warnings
warnings.filterwarnings('ignore')

class VolatilityAnalyzer:
    def __init__(self, csv_path):
        """
        Initialise l'analyseur de volatilité
        
        Args:
            csv_path (str): Chemin vers le fichier CSV à analyser
        """
        self.csv_path = Path(csv_path)
        self.data = None
        self.results = {}

    def validate_timestamp(self, timestamp):
        """
        Valide si un timestamp en nanosecondes est dans une plage acceptable
        """
        try:
            if not timestamp or str(timestamp).strip() == '':
                return False

            timestamp_str = str(timestamp).strip()
            timestamp_val = float(timestamp_str)
            
            # Conversion en millisecondes selon le format
            if len(timestamp_str.split('.')[0]) >= 16:  # Format nanosecondes
                timestamp_val = timestamp_val / 1_000_000  # nano to milli
            elif len(timestamp_str.split('.')[0]) >= 13:  # Format microsecondes
                timestamp_val = timestamp_val / 1_000  # micro to milli
            elif len(timestamp_str.split('.')[0]) <= 11:  # Format secondes
                timestamp_val *= 1_000  # seconds to milli
                
            # Plage élargie pour accommoder les données
            min_date = pd.Timestamp('1970-01-01').value // 10**6
            max_date = pd.Timestamp('2100-01-01').value // 10**6
            
            return min_date <= timestamp_val <= max_date
            
        except Exception as e:
            return False

    def load_data(self):
        """Charge et traite les données du fichier CSV"""
        try:
            print(f"\nChargement du fichier: {self.csv_path}")
            
            if not self.csv_path.exists():
                raise FileNotFoundError(f"Fichier non trouvé: {self.csv_path}")
                
            if self.csv_path.stat().st_size == 0:
                raise ValueError("Fichier vide")
            
            # Lecture du CSV
            self.data = pd.read_csv(
                self.csv_path,
                header=None,
                dtype=str
            )
            
            print("Traitement des données...")
            num_columns = self.data.shape[1]
            
            if num_columns == 1:
                # Si une seule colonne, on essaie de la diviser
                parsed_data = []
                for _, row in self.data.iterrows():
                    try:
                        values = row[0].split(',')
                        if len(values) == 6:
                            parsed_data.append(values)
                    except:
                        continue
                
                self.data = pd.DataFrame(
                    parsed_data,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
            else:
                # Le fichier est déjà en colonnes
                self.data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Conversion des types
            initial_rows = len(self.data)
            
            # Conversion des colonnes de prix et volume
            for col in ['open', 'high', 'low', 'close', 'volume']:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Conversion des timestamps
            # Division par 1M pour convertir nanosecondes en millisecondes
            self.data['datetime'] = pd.to_datetime(
                pd.to_numeric(self.data['timestamp']) / 1_000_000,
                unit='ms'
            )
            
            self.data = self.data.dropna()
            
            # Validation des relations de prix
            self.data = self.data[
                (self.data['high'] >= self.data['low']) &
                (self.data['high'] >= self.data['open']) &
                (self.data['high'] >= self.data['close']) &
                (self.data['low'] <= self.data['open']) &
                (self.data['low'] <= self.data['close']) &
                (self.data['volume'] >= 0)
            ]
            
            dropped_rows = initial_rows - len(self.data)
            
            if len(self.data) == 0:
                raise ValueError("Aucune donnée valide après nettoyage")
            
            # Tri chronologique
            self.data = self.data.sort_values('datetime')
            
            # Calcul des colonnes dérivées
            self.data['returns'] = self.data['close'].pct_change()
            self.data['price_range'] = (self.data['high'] - self.data['low']) / self.data['close']
            
            print(f"Lignes traitées: {initial_rows:,}")
            if dropped_rows > 0:
                print(f"Lignes ignorées: {dropped_rows:,}")
            print(f"Lignes valides: {len(self.data):,}")
            
            print("\nAperçu des données:")
            print(self.data.head())
            
        except Exception as e:
            print(f"\nErreur détaillée lors du chargement:")
            import traceback
            traceback.print_exc()
            raise Exception(f"Erreur lors du chargement des données: {str(e)}")

    def calculate_basic_metrics(self):
        """Calcule les métriques de base de volatilité"""
        try:
            if len(self.data) == 0:
                raise ValueError("Pas de données à analyser")
                
            returns = self.data['returns'].dropna()
            
            # Volatilité historique sur différentes fenêtres
            windows = [5, 10, 20, 30]
            volatilities = {
                f'volatility_{w}d': returns.rolling(window=w).std() * np.sqrt(252)
                for w in windows
            }
            
            # True Range et ATR
            true_range = pd.DataFrame({
                'hl': self.data['high'] - self.data['low'],
                'hc': abs(self.data['high'] - self.data['close'].shift()),
                'lc': abs(self.data['low'] - self.data['close'].shift())
            }).max(axis=1)
            
            atr_windows = [5, 14, 30]
            atrs = {
                f'atr_{w}': true_range.rolling(window=w, min_periods=1).mean()
                for w in atr_windows
            }
            
            self.results['volatility'] = {
                'daily_volatility': returns.std() * np.sqrt(252),
                **{k: v.iloc[-1] for k, v in volatilities.items()},
                **{k: v.iloc[-1] for k, v in atrs.items()},
                'max_daily_range': self.data['price_range'].max(),
                'avg_daily_range': self.data['price_range'].mean()
            }
            
        except Exception as e:
            raise Exception(f"Erreur lors du calcul des métriques de base: {str(e)}")
        
    def calculate_advanced_metrics(self):
        """Calcule des métriques avancées de risque"""
        try:
            if len(self.data) == 0:
                raise ValueError("Pas de données à analyser")
                
            returns = self.data['returns'].dropna()
            
            if len(returns) < 2:
                raise ValueError("Données insuffisantes pour calculer les métriques avancées")
            
            # Calcul du drawdown maximum
            prices = self.data['close']
            peak = prices.expanding().max()
            drawdown = (prices - peak) / peak
            max_drawdown = abs(drawdown.min())
            
            # Calcul des VaR et CVaR pour différents niveaux de confiance
            confidence_levels = [0.99, 0.95, 0.90]
            var_metrics = {
                f'var_{int(c*100)}': np.percentile(returns, (1-c)*100)
                for c in confidence_levels
            }
            
            cvar_metrics = {
                f'cvar_{int(c*100)}': returns[returns <= np.percentile(returns, (1-c)*100)].mean()
                for c in confidence_levels
            }
            
            self.results['advanced_metrics'] = {
                **var_metrics,
                **cvar_metrics,
                'max_drawdown': max_drawdown,
                'kurtosis': stats.kurtosis(returns, nan_policy='omit'),
                'skewness': stats.skew(returns, nan_policy='omit')
            }
            
        except Exception as e:
            raise Exception(f"Erreur lors du calcul des métriques avancées: {str(e)}")

    def generate_report(self):
        """Génère un rapport détaillé d'analyse"""
        try:
            if len(self.data) == 0:
                return "Pas de données disponibles pour générer le rapport"
                
            report = []
            report.append("RAPPORT D'ANALYSE DE VOLATILITÉ")
            report.append("=" * 50)
            
            # Informations générales
            start_date = self.data['datetime'].min()
            end_date = self.data['datetime'].max()
            duration_days = (end_date - start_date).days
            
            report.append(f"\nINFORMATIONS")
            report.append("-" * 20)
            report.append(f"Fichier: {self.csv_path.name}")
            report.append(f"Période: du {start_date.strftime('%d/%m/%Y')} au {end_date.strftime('%d/%m/%Y')}")
            report.append(f"Durée: {duration_days} jours")
            report.append(f"Nombre d'observations: {len(self.data):,}")
            
            # Prix
            report.append("\nPRIX")
            report.append("-" * 20)
            report.append(f"Premier: {self.data['close'].iloc[0]:.4f}")
            report.append(f"Dernier: {self.data['close'].iloc[-1]:.4f}")
            report.append(f"Plus haut: {self.data['high'].max():.4f}")
            report.append(f"Plus bas: {self.data['low'].min():.4f}")
            variation = ((self.data['close'].iloc[-1] / self.data['close'].iloc[0]) - 1) * 100
            report.append(f"Variation totale: {variation:.2f}%")
            
            # Volume
            report.append("\nVOLUME")
            report.append("-" * 20)
            report.append(f"Volume total: {self.data['volume'].sum():.0f}")
            report.append(f"Volume moyen: {self.data['volume'].mean():.2f}")
            report.append(f"Volume médian: {self.data['volume'].median():.2f}")
            report.append(f"Volume max: {self.data['volume'].max():.2f}")
            
            # Volatilité
            vol = self.results['volatility']
            report.append("\nVOLATILITÉ")
            report.append("-" * 20)
            report.append(f"Volatilité annualisée: {vol['daily_volatility']*100:.2f}%")
            
            for w in [5, 10, 20, 30]:
                report.append(f"Volatilité {w}j: {vol[f'volatility_{w}d']*100:.2f}%")
            
            for w in [5, 14, 30]:
                report.append(f"ATR {w}j: {vol[f'atr_{w}']:.4f}")
            
            report.append(f"Range moyen: {vol['avg_daily_range']*100:.2f}%")
            report.append(f"Range maximum: {vol['max_daily_range']*100:.2f}%")
            
            # Risque
            adv = self.results['advanced_metrics']
            report.append("\nRISQUE")
            report.append("-" * 20)
            
            for conf in [90, 95, 99]:
                report.append(f"VaR {conf}%: {adv[f'var_{conf}']*100:.2f}%")
                report.append(f"CVaR {conf}%: {adv[f'cvar_{conf}']*100:.2f}%")
            
            report.append(f"Drawdown max: {adv['max_drawdown']*100:.2f}%")
            
            # Distribution des rendements
            returns = self.data['returns'].dropna()
            report.append("\nDISTRIBUTION DES RENDEMENTS")
            report.append("-" * 20)
            report.append(f"Rendement moyen: {returns.mean()*100:.4f}%")
            report.append(f"Rendement médian: {returns.median()*100:.4f}%")
            report.append(f"Kurtosis: {adv['kurtosis']:.2f}")
            report.append(f"Skewness: {adv['skewness']:.2f}")
            report.append(f"Meilleur rendement: {returns.max()*100:.2f}%")
            report.append(f"Pire rendement: {returns.min()*100:.2f}%")
            
            pos_days = (returns > 0).sum()
            neg_days = (returns < 0).sum()
            total_days = len(returns)
            
            report.append(f"Rendements positifs: {pos_days:,} ({pos_days/total_days*100:.1f}%)")
            report.append(f"Rendements négatifs: {neg_days:,} ({neg_days/total_days*100:.1f}%)")
            
            return "\n".join(report)
            
        except Exception as e:
            raise Exception(f"Erreur lors de la génération du rapport: {str(e)}")
        
    def analyze(self):
        """Effectue l'analyse complète"""
        print("\nDébut de l'analyse...")
        self.load_data()
        
        print("\nCalcul des métriques de base...")
        self.calculate_basic_metrics()
        
        print("Calcul des métriques avancées...")
        self.calculate_advanced_metrics()
        
        print("Génération du rapport...")
        report = self.generate_report()
        
        # Sauvegarde du rapport
        output_path = self.csv_path.parent / f"rapport_volatilite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nRapport sauvegardé: {output_path}")
        print("\nRAPPORT:")
        print(report)

def select_file():
    """Ouvre une boîte de dialogue pour sélectionner le fichier CSV"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Sélectionnez le fichier CSV à analyser",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    return file_path if file_path else None

def main():
    print("Analyse de Volatilité")
    print("=" * 50)
    
    try:
        csv_path = select_file()
        if not csv_path:
            print("Aucun fichier sélectionné. Arrêt du programme.")
            return
        
        analyzer = VolatilityAnalyzer(csv_path)
        analyzer.analyze()
        
    except Exception as e:
        print(f"\nErreur lors de l'analyse: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nAnalyse terminée")

if __name__ == "__main__":
    main()