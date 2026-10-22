"""Local exploratory analysis; no data is transmitted."""
import numpy as np
import pandas as pd
import optuna
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency, false_discovery_control
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             confusion_matrix, roc_curve, balanced_accuracy_score)
from sklearn.inspection import permutation_importance

FEATURES = ['age_years','ef_percent','creatinine_value','total_cholesterol','ldl_mmol_l',
            'glucose_value','crp_value','egfr_value','bmi','female','prior_vascular_surgery',
            'mi_history','stroke_history','vascular_group']
NUMERIC = FEATURES[:9]

def read_data(path):
    if str(path).lower().endswith(('.xlsx', '.xls')):
        frame = pd.read_excel(path)
    else:
        try:
            frame = pd.read_csv(path, sep=None, engine='python')
        except UnicodeDecodeError:
            frame = pd.read_csv(path, sep=None, engine='python', encoding='cp1251')
    frame = frame.replace(r'^\s*$', np.nan, regex=True).dropna(how='all').dropna(axis=1, how='all')
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.columns.duplicated().any():
        raise ValueError('Повторяющиеся названия столбцов. Исправьте заголовки.')
    return frame.loc[:, ~frame.columns.str.startswith('Unnamed')].reset_index(drop=True)

def statistics(x, y, categorical):
    rows = []
    for name in x:
        good = x[name].notna()
        a, b = x.loc[good & y.eq(0), name], x.loc[good & y.eq(1), name]
        p, effect, test = np.nan, np.nan, 'Недостаточно данных'
        if len(a) and len(b) and x.loc[good, name].nunique() > 1:
            if name in categorical:
                tab = pd.crosstab(x.loc[good, name], y[good]).reindex(columns=[0,1], fill_value=0)
                expected = chi2_contingency(tab, correction=False).expected_freq
                if tab.shape == (2,2):
                    effect, p = fisher_exact(tab)
                    test = 'Фишер; ОШ (порядок категорий таблицы)'
                elif (expected >= 5).all():
                    _, p, _, _ = chi2_contingency(tab, correction=False)
                    test = 'χ²'
                else:
                    test = 'Разреженная таблица >2 категорий: p не рассчитано'
            else:
                u, p = mannwhitneyu(b, a, alternative='two-sided', method='auto')
                effect = 2*u/(len(a)*len(b))-1
                test = 'Манна–Уитни; r_rb'
        rows.append({'Признак':name,'n₀':len(a),'n₁':len(b),'Метод':test,'Эффект':effect,'p':p})
    result = pd.DataFrame(rows)
    valid = result.p.notna()
    result['q (FDR)'] = np.nan
    result.loc[valid, 'q (FDR)'] = false_discovery_control(result.loc[valid,'p'])
    return result

def splits(x, y, groups, count, seed):
    if min(pd.Series(y).value_counts()) < count:
        raise ValueError(f'Для {count} блоков недостаточно записей одного из исходов.')
    result = list(StratifiedGroupKFold(count, shuffle=True, random_state=seed).split(x,y,groups))
    for train, test in result:
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError('В одном из групповых блоков нет обоих исходов. Уменьшите число блоков или используйте общую выборку.')
        assert not set(groups[train]) & set(groups[test])
    return result

def pipeline(x, categorical, family, params):
    numeric = [c for c in x if c not in categorical]
    pre = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True), StandardScaler()), numeric),
        ('cat', make_pipeline(SimpleImputer(strategy='most_frequent', keep_empty_features=True),
                              OneHotEncoder(handle_unknown='ignore', sparse_output=False)), categorical)])
    model = (RandomForestClassifier(random_state=2026, n_jobs=2, **params) if family == 'RF'
             else LogisticRegression(random_state=2026, max_iter=3000, **params))
    return make_pipeline(pre, model)

def run(frame, features, categorical, target, positive, group_column, family, trials, folds, progress, cancelled):
    raw_y = pd.to_numeric(frame[target], errors='coerce')
    if not raw_y.dropna().isin([0,1]).all():
        raise ValueError('Исход должен содержать только 0, 1 и пропуски. Даты событий сначала преобразуйте в 1.')
    valid = raw_y.notna()
    d = frame.loc[valid].copy()
    if not features or target in features or group_column in features:
        raise ValueError('Выберите предикторы без исхода и идентификатора пациента.')
    x = d[features].copy()
    for c in features:
        if c in categorical:
            x[c] = x[c].map(lambda v: str(v) if pd.notna(v) else np.nan)
        else:
            original = x[c].notna()
            x[c] = pd.to_numeric(x[c].astype('string').str.replace(',', '.', regex=False), errors='coerce').astype(float)
            if (original & x[c].isna()).any() or np.isinf(x[c]).any():
                raise ValueError(f'{c}: нечисловые/бесконечные значения. Исправьте данные или отметьте признак категориальным.')
    y = raw_y[valid].eq(positive).astype(int).to_numpy()
    if group_column:
        if d[group_column].isna().any():
            raise ValueError('Идентификатор пациента не должен содержать пропуски.')
        groups = d[group_column].astype(str).to_numpy()
    else:
        groups = np.arange(len(d))
    stat = statistics(x.reset_index(drop=True), pd.Series(y), categorical)
    outer = splits(x,y,groups,folds,2026)
    predictions, decisions = np.zeros(len(y)), np.zeros(len(y), dtype=int)
    importances, configs = [], []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for k,(train,test) in enumerate(outer):
        if cancelled(): raise InterruptedError('Расчёт отменён')
        a,b,g = x.iloc[train],y[train],groups[train]
        inner = splits(a,b,g,folds,2027+k)
        def objective(trial):
            if cancelled(): raise InterruptedError('Расчёт отменён')
            if family == 'RF':
                params = dict(n_estimators=trial.suggest_int('n_estimators',200,800,step=100),
                              max_depth=trial.suggest_int('max_depth',2,8),
                              min_samples_leaf=trial.suggest_categorical('min_samples_leaf',[1,2,4,8,16]),
                              max_features=trial.suggest_categorical('max_features',['sqrt',0.5,1.0]))
            else:
                params = {'C':trial.suggest_float('C',.001,100,log=True)}
            scores=[]
            for tr,va in inner:
                if cancelled(): raise InterruptedError('Расчёт отменён')
                model=pipeline(a,categorical,family,params).fit(a.iloc[tr],b[tr])
                scores.append(average_precision_score(b[va],model.predict_proba(a.iloc[va])[:,1]))
            progress(f'Внешний блок {k+1}/{folds}, Optuna {trial.number+1}/{trials}')
            return np.mean(scores)
        study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=2026+k))
        study.optimize(objective,n_trials=trials)
        inner_p=np.zeros(len(b))
        for tr,va in inner:
            model=pipeline(a,categorical,family,study.best_params).fit(a.iloc[tr],b[tr])
            inner_p[va]=model.predict_proba(a.iloc[va])[:,1]
        fpr,tpr,thresholds=roc_curve(b,inner_p)
        eligible=np.isfinite(thresholds)
        threshold=thresholds[eligible][np.argmax((tpr-fpr)[eligible])]
        model=pipeline(a,categorical,family,study.best_params).fit(a,b)
        predictions[test]=model.predict_proba(x.iloc[test])[:,1]
        decisions[test]=predictions[test]>=threshold
        imp=permutation_importance(model,x.iloc[test],y[test],scoring='average_precision',n_repeats=5,random_state=2026)
        importances.append(imp.importances_mean)
        configs.append({'fold':k+1,'params':study.best_params,'threshold':float(threshold),'inner_AP':study.best_value})
    tn,fp,fn,tp=confusion_matrix(y,decisions,labels=[0,1]).ravel()
    metrics={'N':len(y),'Событий':int(y.sum()),'Исключено без исхода':int((~valid).sum()),
             'AP':average_precision_score(y,predictions),'ROC AUC':roc_auc_score(y,predictions),
             'Brier':brier_score_loss(y,predictions),'Recall события':tp/(tp+fn),
             'Специфичность':tn/(tn+fp),'Precision':tp/(tp+fp) if tp+fp else 0,
             'Balanced accuracy':balanced_accuracy_score(y,decisions)}
    return dict(metrics=metrics,stats=stat,importance=pd.DataFrame({'Признак':features,'ΔAP':np.mean(importances,axis=0)}),
                predictions=pd.DataFrame({'row_index':d.index,'outcome':y,'probability':predictions,'prediction':decisions}),
                configs=configs,features=features,categorical=categorical,family=family,folds=folds,trials=trials,
                target=target,positive=positive,group_column=group_column)
