"""
Experiment 1: Cross-Author Transfer
Train on Dickens (Human vs AI Impostor), test on Austen (Human vs AI Impostor).
If accuracy stays high, the detector finds AI fingerprints, not author fingerprints.
Experiment 2: Feature Ablation
Remove one feature group at a time and measure accuracy drop.
"""
importsys
importjson
importnumpyasnp
importpandasaspd
frompathlibimportPath
fromsklearn.ensembleimportRandomForestClassifier
fromsklearn.metricsimportclassification_report,accuracy_score,roc_auc_score
fromsklearn.preprocessingimportlabel_binarize
sys.path.insert(0,str(Path(__file__).parent))
fromsrc.configimportCFG,PROJECT_ROOT
FUNC_WORDS=["the","of","and","to","a","in","that","it","is","was",
"for","on","as","with","his","he","be","at","by","this"]
FEATURE_GROUPS={
"Lexical Richness":["ttr","n_tokens","n_types"],
"Syntax":["adj_noun_ratio","avg_dep_depth","n_sentences","avg_sentence_len"],
"Readability":["flesch_kincaid"],
"Punctuation":[f"punct_{p}"forpin[";","—","!","?",",",".",":","(",")"]],
"Function Words":[f"func_{fw}"forfwinFUNC_WORDS],
}
ALL_FEATS=[]
forgroup_featsinFEATURE_GROUPS.values():
    ALL_FEATS.extend(group_feats)
RESULTS={}
defload_data():
    feats=pd.read_csv(PROJECT_ROOT/"data"/"processed"/"features.csv")
feats["binary_label"]=feats["label"].map({"Human":0,"AI_Generic":1,"AI_Impostor":1})
returnfeats
defexperiment_cross_author(feats):
    print("="*60)
print("EXPERIMENT 1: CROSS-AUTHOR TRANSFER")
print("="*60)
dickens_human=feats[(feats["label"]=="Human")&(feats["author"]=="Dickens")]
dickens_ai=feats[(feats["label"]=="AI_Impostor")&(feats["author"]=="Dickens")]
austen_human=feats[(feats["label"]=="Human")&(feats["author"]=="Austen")]
austen_ai=feats[(feats["label"]=="AI_Impostor")&(feats["author"]=="Austen")]
train_df=pd.concat([dickens_human,dickens_ai],ignore_index=True)
test_df=pd.concat([austen_human,austen_ai],ignore_index=True)
print(f"\nTrain: {len(dickens_human)} Dickens Human + {len(dickens_ai)} Dickens AI = {len(train_df)}")
print(f"Test:  {len(austen_human)} Austen Human + {len(austen_ai)} Austen AI = {len(test_df)}")
X_train=train_df[ALL_FEATS].values
y_train=train_df["binary_label"].values
X_test=test_df[ALL_FEATS].values
y_test=test_df["binary_label"].values
clf=RandomForestClassifier(n_estimators=400,class_weight="balanced",random_state=42,n_jobs=-1)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
y_proba=clf.predict_proba(X_test)[:,1]
acc=accuracy_score(y_test,y_pred)
auc=roc_auc_score(y_test,y_proba)
print(f"\nAccuracy: {acc:.3f}")
print(f"AUC: {auc:.3f}")
print(f"\n{classification_report(y_test,y_pred,target_names=['Human','AI'])}")
reverse_train=pd.concat([austen_human,austen_ai],ignore_index=True)
reverse_test=pd.concat([dickens_human,dickens_ai],ignore_index=True)
clf2=RandomForestClassifier(n_estimators=400,class_weight="balanced",random_state=42,n_jobs=-1)
clf2.fit(reverse_train[ALL_FEATS].values,reverse_train["binary_label"].values)
y_pred2=clf2.predict(reverse_test[ALL_FEATS].values)
y_proba2=clf2.predict_proba(reverse_test[ALL_FEATS].values)[:,1]
acc2=accuracy_score(reverse_test["binary_label"].values,y_pred2)
auc2=roc_auc_score(reverse_test["binary_label"].values,y_proba2)
print(f"Reverse (Train Austen, Test Dickens):")
print(f"  Accuracy: {acc2:.3f}")
print(f"  AUC: {auc2:.3f}")
print(f"\n{classification_report(reverse_test['binary_label'].values,y_pred2,target_names=['Human','AI'])}")
RESULTS["cross_author"]={
"dickens_to_austen":{"accuracy":round(acc,3),"auc":round(auc,3)},
"austen_to_dickens":{"accuracy":round(acc2,3),"auc":round(auc2,3)},
}
defexperiment_feature_ablation(feats):
    print("="*60)
print("EXPERIMENT 2: FEATURE ABLATION")
print("="*60)
train_csv=pd.read_csv(PROJECT_ROOT/"data"/"processed"/"train.csv")
test_csv=pd.read_csv(PROJECT_ROOT/"data"/"processed"/"test.csv")
train_ids=set(train_csv["sample_id"].tolist())
test_ids=set(test_csv["sample_id"].tolist())
train_df=feats[feats["sample_id"].isin(train_ids)]
test_df=feats[feats["sample_id"].isin(test_ids)]
label_map={"Human":0,"AI_Generic":1,"AI_Impostor":2}
y_train=train_df["label"].map(label_map).values
y_test=test_df["label"].map(label_map).values
clf_all=RandomForestClassifier(n_estimators=400,class_weight="balanced",random_state=42,n_jobs=-1)
clf_all.fit(train_df[ALL_FEATS].values,y_train)
baseline_acc=accuracy_score(y_test,clf_all.predict(test_df[ALL_FEATS].values))
y_proba_all=clf_all.predict_proba(test_df[ALL_FEATS].values)
y_test_bin=label_binarize(y_test,classes=[0,1,2])
baseline_auc=roc_auc_score(y_test_bin,y_proba_all,average="macro",multi_class="ovr")
print(f"\nBaseline (all features): Accuracy={baseline_acc:.3f}, AUC={baseline_auc:.3f}")
print(f"Total features: {len(ALL_FEATS)}\n")
ablation_results={"baseline":{"accuracy":round(baseline_acc,3),"auc":round(baseline_auc,3),"n_features":len(ALL_FEATS)}}
forgroup_name,group_featsinFEATURE_GROUPS.items():
        remaining=[fforfinALL_FEATSiffnotingroup_feats]
clf=RandomForestClassifier(n_estimators=400,class_weight="balanced",random_state=42,n_jobs=-1)
clf.fit(train_df[remaining].values,y_train)
acc=accuracy_score(y_test,clf.predict(test_df[remaining].values))
y_proba=clf.predict_proba(test_df[remaining].values)
auc=roc_auc_score(y_test_bin,y_proba,average="macro",multi_class="ovr")
drop=baseline_acc-acc
print(f"  Without {group_name:20s} ({len(group_feats):2d} feats): Acc={acc:.3f} (Δ={drop:+.3f}), AUC={auc:.3f}")
ablation_results[group_name]={
"accuracy":round(acc,3),
"auc":round(auc,3),
"accuracy_drop":round(drop,3),
"n_features_removed":len(group_feats),
"n_features_remaining":len(remaining),
}
print()
forgroup_name,group_featsinFEATURE_GROUPS.items():
        clf=RandomForestClassifier(n_estimators=400,class_weight="balanced",random_state=42,n_jobs=-1)
clf.fit(train_df[group_feats].values,y_train)
acc=accuracy_score(y_test,clf.predict(test_df[group_feats].values))
y_proba=clf.predict_proba(test_df[group_feats].values)
auc=roc_auc_score(y_test_bin,y_proba,average="macro",multi_class="ovr")
print(f"  ONLY {group_name:20s} ({len(group_feats):2d} feats): Acc={acc:.3f}, AUC={auc:.3f}")
ablation_results[f"only_{group_name}"]={
"accuracy":round(acc,3),
"auc":round(auc,3),
"n_features":len(group_feats),
}
RESULTS["feature_ablation"]=ablation_results
defmain():
    feats=load_data()
experiment_cross_author(feats)
print()
experiment_feature_ablation(feats)
log_path=PROJECT_ROOT/"results"/"experiment_logs.json"
withopen(log_path,"w")asf:
        json.dump(RESULTS,f,indent=2)
print(f"\nResults saved to {log_path}")
if__name__=="__main__":
    main()