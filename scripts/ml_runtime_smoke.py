from __future__ import annotations
import json,platform
from pathlib import Path
import lightgbm as lgb
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression,Ridge

def run():
 rng=np.random.default_rng(20260712); X=rng.normal(size=(200,6)); y=X[:,0]-.5*X[:,1]+rng.normal(0,.1,200); c=(y>np.median(y)).astype(int)
 ridge=Ridge(alpha=1).fit(X,y); logit=LogisticRegression(C=1,solver='liblinear',random_state=20260712).fit(X,c); tree=lgb.LGBMRegressor(n_estimators=30,num_leaves=7,learning_rate=.05,n_jobs=1,random_state=20260712,deterministic=True,verbosity=-1).fit(X,y)
 result={'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),'sklearn':sklearn.__version__,'lightgbm':lgb.__version__,'seed':20260712,'threads':1,'ridge_checksum':float(np.round(ridge.predict(X[:10]).sum(),12)),'logit_checksum':float(np.round(logit.predict_proba(X[:10])[:,1].sum(),12)),'lightgbm_checksum':float(np.round(tree.predict(X[:10]).sum(),12))}
 out=Path('outputs/vnext_p3_ml_runtime_20260712'); out.mkdir(parents=True,exist_ok=True); (out/'deterministic_smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result))
if __name__=='__main__':run()
