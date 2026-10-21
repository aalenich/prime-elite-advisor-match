from pathlib import Path
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, ndcg_score
import joblib
ROOT=Path(__file__).resolve().parent
SKILLS=['bonds','ru_equities','global_markets','deposits','lending','margin','structured_products']
COLS=['region_match','channel_match','language_match','hobby_overlap','load_ratio','experience_years','capital_mln','intensity','investment_experience','need_count']+[f'need_{s}' for s in SKILLS]+[f'skill_{s}' for s in SKILLS]
CATS=['intensity','investment_experience']
def features(df):
    x=pd.DataFrame(index=df.index)
    x['region_match']=(df.client_region==df.manager_region).astype(int)
    x['channel_match']=(df.client_channel==df.manager_channel).astype(int)
    x['language_match']=[int(bool(set(str(a).split('_')) & set(str(b).split('_')))) for a,b in zip(df.client_languages,df.manager_languages)]
    x['hobby_overlap']=[len(set(str(a).split('|')) & set(str(b).split('|'))) for a,b in zip(df.client_hobbies,df.manager_hobbies)]
    x['load_ratio']=df.manager_initial_clients / df.manager_capacity
    x['experience_years']=df.manager_experience_years
    x['capital_mln']=df.client_capital_mln_rub
    x['intensity']=df.service_intensity.fillna('medium')
    x['investment_experience']=df.investment_experience.fillna('beginner')
    x['need_count']=[len(str(a).split('|')) for a in df.needs]
    for s in SKILLS:
        x[f'need_{s}']=[int(s in str(a).split('|')) for a in df.needs]
        x[f'skill_{s}']=df[f'skill_{s}']
    return x[COLS]
def baseline(df):
    need_score=[]
    for _,r in df.iterrows():
        ns=str(r.needs).split('|');need_score.append(np.mean([r['skill_'+s] for s in ns if 'skill_'+s in r.index])/10 if ns else 0)
    return np.clip(100*(.45*np.array(need_score)+.2*(df.client_region==df.manager_region).astype(int)+.2*(1-df.manager_initial_clients/df.manager_capacity)+.1*(df.client_channel==df.manager_channel).astype(int)+.05*np.array([int(bool(set(str(a).split('|'))&set(str(b).split('|')))) for a,b in zip(df.client_hobbies,df.manager_hobbies)])),0,100)
def main():
    df=pd.read_csv(ROOT/'data/training_pairs.csv')
    x=features(df);y=df.match_score
    tr,te=next(GroupShuffleSplit(test_size=.2,n_splits=1,random_state=42).split(x,y,groups=df.client_id))
    model=CatBoostRegressor(iterations=260,depth=6,learning_rate=.09,loss_function='RMSE',verbose=False,random_seed=42,thread_count=4,l2_leaf_reg=4)
    model.fit(x.iloc[tr],y.iloc[tr],cat_features=CATS,eval_set=(x.iloc[te],y.iloc[te]),early_stopping_rounds=35,verbose=False)
    pred=model.predict(x.iloc[te]);b=baseline(df.iloc[te]); metrics={'test_pairs':len(te),'test_clients':int(df.iloc[te].client_id.nunique()),'mae_ml':round(float(mean_absolute_error(y.iloc[te],pred)),3),'mae_baseline':round(float(mean_absolute_error(y.iloc[te],b)),3),'rmse_ml':round(float(np.sqrt(mean_squared_error(y.iloc[te],pred))),3),'r2_ml':round(float(r2_score(y.iloc[te],pred)),3),'iterations':model.tree_count_}
    ranked=pd.DataFrame({'client_id':df.iloc[te].client_id.values,'truth':y.iloc[te].values,'prediction':pred,'baseline':b})
    for col in ['prediction','baseline']:
        scores=[]
        for _,g in ranked.groupby('client_id'):
            if len(g)>1:scores.append(ndcg_score(g[['truth']].values.T,g[[col]].values.T,k=3))
        metrics['ndcg3_'+('ml' if col=='prediction' else 'baseline')]=round(float(np.mean(scores)),4)
    model.save_model(str(ROOT/'model/matcher.cbm'))
    (ROOT/'model/metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    imp=sorted(zip(COLS,model.feature_importances_.tolist()),key=lambda z:-z[1]);(ROOT/'model/importance.json').write_text(json.dumps(imp,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
