"""
Task 2 — Tier A: The Statistician.
Random Forest on stylometric features. 
Reads pre-split CSVs generated in Task 0 to ensure strict boundary alignment.
"""
from__future__importannotations
frompathlibimportPath
importnumpyasnp
importpandasaspd
importmatplotlib.pyplotasplt
importshap
fromsklearn.ensembleimportRandomForestClassifier
fromsklearn.model_selectionimportStratifiedKFold,cross_val_score
fromsklearn.metricsimportclassification_report,confusion_matrix,roc_curve,auc
fromsklearn.preprocessingimportlabel_binarize
fromseabornimportheatmapassns_heatmap
importjoblib
from.configimportCFG,PROJECT_ROOT
from.utilsimportload_csv
FUNC_WORDS=["the","of","and","to","a","in","that","it","is","was","for","on","as","with","his","he","be","at","by","this"]
FEAT_COLS=[
"ttr","n_tokens","n_types","adj_noun_ratio","avg_dep_depth",
"flesch_kincaid","n_sentences","avg_sentence_len",
]+[f"punct_{p}"forpin[";","—","!","?",",",".",":","(",")"]]+[f"func_{fw}"forfwinFUNC_WORDS]
LABEL_MAP={"Human":0,"AI_Generic":1,"AI_Impostor":2}
defload_splits_with_features():
    train=load_csv("train.csv")
val=load_csv("val.csv")
test=load_csv("test.csv")
feats=pd.read_csv(PROJECT_ROOT/CFG["paths"]["processed_dir"]/"features.csv")
train=train.merge(feats,on="sample_id",suffixes=("","_feat"))
val=val.merge(feats,on="sample_id",suffixes=("","_feat"))
test=test.merge(feats,on="sample_id",suffixes=("","_feat"))
returntrain,val,test
deffit_and_eval():
    train,val,test=load_splits_with_features()
X_tr,y_tr=train[FEAT_COLS].values,train["label"].map(LABEL_MAP).values
X_test,y_test=test[FEAT_COLS].values,test["label"].map(LABEL_MAP).values
clf=RandomForestClassifier(
n_estimators=CFG["tier_a"]["n_estimators"],
max_depth=CFG["tier_a"]["max_depth"],
class_weight=CFG["tier_a"]["class_weight"],
random_state=42,n_jobs=-1
)
skf=StratifiedKFold(n_splits=CFG["tier_a"]["cv_folds"],shuffle=True,random_state=42)
scores=cross_val_score(clf,X_tr,y_tr,cv=skf,scoring="accuracy")
print(f"[Tier A] 5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
clf.fit(X_tr,y_tr)
y_pred=clf.predict(X_test)
print("\n[Tier A] Test classification report:")
print(classification_report(y_test,y_pred,target_names=list(LABEL_MAP.keys())))
y_test_bin=label_binarize(y_test,classes=[0,1,2])
y_proba=clf.predict_proba(X_test)
fpr,tpr,roc_auc={},{},{}
fori,nameinenumerate(LABEL_MAP.keys()):
        fpr[i],tpr[i],_=roc_curve(y_test_bin[:,i],y_proba[:,i])
roc_auc[i]=auc(fpr[i],tpr[i])
macro_auc=np.mean(list(roc_auc.values()))
fig_dir=PROJECT_ROOT/CFG["paths"]["figures_dir"]
fig_dir.mkdir(parents=True,exist_ok=True)
plt.figure(figsize=(6,5))
fori,nameinenumerate(LABEL_MAP.keys()):
        plt.plot(fpr[i],tpr[i],label=f"{name} (AUC={roc_auc[i]:.3f})")
plt.plot([0,1],[0,1],"k--",alpha=.4)
plt.title(f"Tier A — One-vs-Rest ROC (macro AUC={macro_auc:.3f})")
plt.xlabel("False Positive Rate");plt.ylabel("True Positive Rate")
plt.legend();plt.tight_layout()
plt.savefig(fig_dir/"tier_a_roc.png",dpi=150);plt.close()
explainer=shap.TreeExplainer(clf)
shap_values=explainer.shap_values(X_test)
ifisinstance(shap_values,list):
        sv_arr=np.stack(shap_values,axis=-1)
else:
        sv_arr=shap_values
shap.summary_plot(sv_arr.mean(axis=-1),feature_names=FEAT_COLS,features=X_test,show=False)
plt.tight_layout()
plt.savefig(fig_dir/"tier_a_shap_summary.png",dpi=150,bbox_inches="tight")
plt.close()
model_dir=PROJECT_ROOT/CFG["paths"]["models_dir"]
model_dir.mkdir(parents=True,exist_ok=True)
joblib.dump(clf,model_dir/"tier_a_rf.joblib")
print(f"[Tier A] Test macro AUC = {macro_auc:.3f}")
if__name__=="__main__":
    fit_and_eval()