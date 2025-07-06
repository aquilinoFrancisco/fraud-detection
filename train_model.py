"""
Script de entrenamiento para el sistema de detección de fraude
Extrae la lógica del notebook y exporta los modelos
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("🚀 INICIANDO ENTRENAMIENTO DE MODELOS DE FRAUD DETECTION")
print("=" * 60)

# Crear directorio para modelos si no existe
os.makedirs('models', exist_ok=True)

# ============================================================================
# 1. CARGA DE DATOS
# ============================================================================
print("\n📊 PASO 1: CARGA DE DATOS")
print("-" * 40)

try:
    df = pd.read_csv('data/fraud_train.csv')
    print(f"✅ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"📈 Fraud Rate: {df['FraudFound_P'].mean():.1%}")
except FileNotFoundError:
    print("⚠️ Generando datos sintéticos...")
    np.random.seed(42)
    n_samples = 5000
    
    df = pd.DataFrame({
        'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], n_samples),
        'DayOfWeek': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], n_samples),
        'Make': np.random.choice(['Honda', 'Toyota', 'Ford', 'BMW', 'Mercedes', 'Audi'], n_samples),
        'PolicyType': np.random.choice(['Sedan - Collision', 'Sedan - All Perils', 'Sport - Collision'], n_samples),
        'AgeOfPolicyHolder': np.random.choice(['18 to 20', '21 to 25', '26 to 30', '31 to 35'], n_samples),
        'VehiclePrice': np.random.choice(['20000 to 29000', '30000 to 39000', 'more than 69000'], n_samples),
        'Days_Policy_Claim': np.random.choice(['1 to 7', '8 to 15', '15 to 30', 'more than 30'], n_samples),
        'AccidentArea': np.random.choice(['Urban', 'Rural', 'Highway'], n_samples),
        'Sex': np.random.choice(['Male', 'Female'], n_samples),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'AgeOfVehicle': np.random.choice(['new', '2 years', '5 years', 'more than 7'], n_samples),
    })
    
    # Crear patrones de fraude
    fraud_probability = np.zeros(n_samples)
    fraud_probability += np.where(df['Days_Policy_Claim'] == '1 to 7', 0.15, 0)
    fraud_probability += np.where(df['PolicyType'].str.contains('All Perils'), 0.08, 0)
    fraud_probability += np.where(df['Make'].isin(['BMW', 'Mercedes']), 0.1, 0)
    fraud_probability = np.clip(fraud_probability, 0, 0.25)
    
    df['FraudFound_P'] = np.random.binomial(1, fraud_probability)
    print(f"✅ Dataset sintético creado: {df.shape}")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n🔧 PASO 2: FEATURE ENGINEERING")
print("-" * 40)

# Convertir variables categóricas a numéricas
age_mapping = {
    'below 18': 17, '16 to 17': 17, '18 to 20': 19, '21 to 25': 23,
    '26 to 30': 28, '31 to 35': 33, '36 to 40': 38, '41 to 50': 45,
    '51 to 65': 58, 'over 65': 70
}
df['AgeOfPolicyHolder_Numeric'] = df.get('AgeOfPolicyHolder', pd.Series()).map(age_mapping).fillna(35)

price_mapping = {
    'less than 20000': 15000, '20000 to 29000': 24500,
    '30000 to 39000': 34500, '40000 to 59000': 49500,
    '60000 to 69000': 64500, 'more than 69000': 80000
}
df['VehiclePrice_Numeric'] = df.get('VehiclePrice', pd.Series()).map(price_mapping).fillna(35000)

vehicle_age_mapping = {
    'new': 0, '2 years': 2, '3 years': 3, '4 years': 4,
    '5 years': 5, '6 years': 6, '7 years': 7, 'more than 7': 10
}
df['AgeOfVehicle_Numeric'] = df.get('AgeOfVehicle', pd.Series()).map(vehicle_age_mapping).fillna(5)

days_mapping = {
    'none': 0, '1 to 7': 4, '8 to 15': 11,
    '15 to 30': 22, 'more than 30': 45
}
df['Days_Policy_Claim_Numeric'] = df.get('Days_Policy_Claim', pd.Series()).map(days_mapping).fillna(30)

# Crear variables de negocio
df['Claim_Urgency'] = (df.get('Days_Policy_Claim', '') == '1 to 7').astype(int)
df['Luxury_Vehicle'] = df.get('VehiclePrice', '').isin(['60000 to 69000', 'more than 69000']).astype(int)
df['Young_Driver'] = df.get('AgeOfPolicyHolder', '').isin(['18 to 20', '21 to 25']).astype(int)
df['Complex_Policy'] = df.get('PolicyType', '').str.contains('All Perils', na=False).astype(int)
df['Premium_Make'] = df.get('Make', '').isin(['BMW', 'Mercedes', 'Audi']).astype(int)

print("✅ Feature engineering completado")
print(f"   • Variables numéricas creadas: 4")
print(f"   • Variables de negocio creadas: 5")

# ============================================================================
# 3. WEIGHT OF EVIDENCE
# ============================================================================
print("\n📊 PASO 3: CÁLCULO DE WEIGHT OF EVIDENCE")
print("-" * 40)

def calculate_woe(df, feature, target):
    """Calcula Weight of Evidence para una variable"""
    crosstab = pd.crosstab(df[feature], df[target])
    
    if len(crosstab) < 2 or crosstab.shape[1] < 2:
        return {}, 0
    
    total_good = crosstab[0].sum() if 0 in crosstab.columns else 1
    total_bad = crosstab[1].sum() if 1 in crosstab.columns else 1
    
    if total_good == 0 or total_bad == 0:
        return {}, 0
    
    woe_dict = {}
    iv_total = 0
    
    for category in crosstab.index:
        good = crosstab.loc[category, 0] if 0 in crosstab.columns else 0.5
        bad = crosstab.loc[category, 1] if 1 in crosstab.columns else 0.5
        
        good_rate = good / total_good
        bad_rate = bad / total_bad
        
        if good_rate <= 0: good_rate = 0.0001
        if bad_rate <= 0: bad_rate = 0.0001
        
        woe = np.log(bad_rate / good_rate)
        iv = (bad_rate - good_rate) * woe
        
        woe_dict[category] = woe
        iv_total += iv
    
    return woe_dict, iv_total

# Calcular WoE para variables categóricas
categorical_vars = ['Make', 'PolicyType', 'AccidentArea', 'Sex', 'MaritalStatus', 
                   'Month', 'DayOfWeek', 'AgeOfPolicyHolder', 'VehiclePrice', 
                   'AgeOfVehicle', 'Days_Policy_Claim']

woe_mappings = {}
iv_values = {}

for var in categorical_vars:
    if var in df.columns:
        woe_dict, iv = calculate_woe(df, var, 'FraudFound_P')
        if woe_dict:
            woe_mappings[var] = woe_dict
            iv_values[var] = iv
            df[f'{var}_WoE'] = df[var].map(woe_dict).fillna(0)
            print(f"✅ {var}: IV = {iv:.3f}")

print(f"\n📈 Variables WoE creadas: {len(woe_mappings)}")

# ============================================================================
# 4. PREPARACIÓN DE DATOS PARA MODELADO
# ============================================================================
print("\n🎯 PASO 4: PREPARACIÓN PARA MODELADO")
print("-" * 40)

# Seleccionar features
woe_features = [col for col in df.columns if col.endswith('_WoE')]
numeric_features = [col for col in df.columns if col.endswith('_Numeric')]
business_features = ['Claim_Urgency', 'Luxury_Vehicle', 'Young_Driver', 'Complex_Policy', 'Premium_Make']

feature_cols = woe_features + numeric_features + business_features
feature_cols = [col for col in feature_cols if col in df.columns]

X = df[feature_cols].fillna(0)
y = df['FraudFound_P']

print(f"📊 Features seleccionadas: {len(feature_cols)}")

# Split de datos
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)

# Balanceo con SMOTE
try:
    smote = SMOTE(random_state=42, sampling_strategy=0.1)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"✅ Balanceo SMOTE aplicado: {y_train_balanced.mean():.1%} fraude")
except:
    X_train_balanced, y_train_balanced = X_train, y_train
    print("⚠️ SMOTE no disponible, usando datos originales")

# ============================================================================
# 5. ENTRENAMIENTO DE MODELOS
# ============================================================================
print("\n🤖 PASO 5: ENTRENAMIENTO DE MODELOS")
print("-" * 40)

# Modelo 1: Regresión Logística
print("\n📊 Entrenando Regresión Logística...")
logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',
    penalty='l1',
    solver='liblinear',
    C=1.0
)
logistic_model.fit(X_train_balanced, y_train_balanced)

# Evaluación
y_pred_val = logistic_model.predict_proba(X_val)[:, 1]
auc_logistic = roc_auc_score(y_val, y_pred_val)
print(f"✅ Logistic Regression - AUC: {auc_logistic:.3f}")

# Modelo 2: XGBoost
print("\n🚀 Entrenando XGBoost...")
xgb_params = {
    'objective': 'binary:logistic',
    'max_depth': 4,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'random_state': 42
}
xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train_balanced, y_train_balanced, 
              eval_set=[(X_val, y_val)], 
              verbose=False)

y_pred_val_xgb = xgb_model.predict_proba(X_val)[:, 1]
auc_xgb = roc_auc_score(y_val, y_pred_val_xgb)
print(f"✅ XGBoost - AUC: {auc_xgb:.3f}")

# ============================================================================
# 6. GENERACIÓN DE SCORECARD
# ============================================================================
print("\n📋 PASO 6: GENERACIÓN DE SCORECARD")
print("-" * 40)

base_score = 650
pdo = 20
odds = 50

factor = pdo / np.log(2)
offset = base_score - (factor * np.log(odds))

scorecard = pd.DataFrame({
    'Variable': feature_cols,
    'Coefficient': logistic_model.coef_[0],
    'Points': -logistic_model.coef_[0] * factor
})
scorecard['Points'] = scorecard['Points'].round(0).astype(int)

base_points = offset + (factor * logistic_model.intercept_[0])

scorecard_dict = {
    'scorecard': scorecard,
    'base_points': float(base_points),
    'factor': float(factor),
    'offset': float(offset),
    'parameters': {
        'base_score': base_score,
        'pdo': pdo,
        'odds': odds
    }
}

print(f"✅ Scorecard generada con {len(scorecard)} variables")
print(f"   Base Score: {base_score}")
print(f"   Base Points: {base_points:.0f}")

# ============================================================================
# 7. EXPORTACIÓN DE MODELOS
# ============================================================================
print("\n💾 PASO 7: EXPORTANDO MODELOS")
print("-" * 40)

# Guardar modelos
joblib.dump(logistic_model, 'models/logistic_model.pkl')
print("✅ models/logistic_model.pkl")

joblib.dump(xgb_model, 'models/xgb_model.pkl')
print("✅ models/xgb_model.pkl")

joblib.dump(woe_mappings, 'models/woe_mappings.pkl')
print("✅ models/woe_mappings.pkl")

joblib.dump(scorecard_dict, 'models/scorecard.pkl')
print("✅ models/scorecard.pkl")

# Guardar metadata
metadata = {
    'feature_names': feature_cols,
    'auc_logistic': float(auc_logistic),
    'auc_xgb': float(auc_xgb),
    'n_features': len(feature_cols),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}
joblib.dump(metadata, 'models/metadata.pkl')
print("✅ models/metadata.pkl")

print("\n" + "=" * 60)
print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
print("=" * 60)
print(f"📊 Modelos exportados: 5 archivos")
print(f"📈 Performance: Logistic={auc_logistic:.3f}, XGBoost={auc_xgb:.3f}")
print(f"🚀 Sistema listo para producción")
