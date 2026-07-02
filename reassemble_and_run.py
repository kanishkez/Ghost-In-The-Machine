"""
Reassemble the full dataset from expanded JSON/book data and re-run the pipeline.
This runs after generate_more_data.py has populated the JSON files.
"""
importjson
importos
importsys
importglob
importrandom
importpandasaspd
importnumpyasnp
frompathlibimportPath
sys.path.insert(0,str(Path(__file__).parent))
fromsrc.utilsimportclean_text,split_into_paragraphs
fromsrc.configimportload_config
TARGET_PER_CLASS=240
DICKENS_IDS=[98,1400,730,766,46]
AUSTEN_IDS=[105,121,1342,158,161]
defbuild_class_a(config,target_per_author=120):
    """Extract human paragraphs from Gutenberg books with book-level tracking."""
raw_dir=Path("data/raw")
all_paras=[]
forauthor_name,book_idsin[("Dickens",DICKENS_IDS),("Austen",AUSTEN_IDS)]:
        author_paras=[]
forbidinbook_ids:
            matches=list(raw_dir.glob(f"{bid}_*.txt"))
ifnotmatches:
                print(f"  [warn] No file found for {author_name} book {bid}")
continue
fpath=matches[0]
text=clean_text(fpath.read_text(encoding="utf-8",errors="ignore"))
paras=split_into_paragraphs(text,100,200)
forpinparas:
                author_paras.append({
"text":p,
"label_class":"A",
"label_author":author_name,
"source_id":f"{author_name}_{bid}",
"book_id":bid,
})
print(f"  {fpath.stem}: {len(paras)} paragraphs")
random.shuffle(author_paras)
selected=author_paras[:target_per_author]
all_paras.extend(selected)
print(f"  → {author_name}: selected {len(selected)} from {len(author_paras)} available\n")
returnall_paras
defbuild_class_b():
    """Load all Class B paragraphs from JSON."""
withopen("data/processed/class_b.json")asf:
        raw=json.load(f)
return[
{"text":x["text"],"label_class":"B","label_author":"None",
"source_id":f"AI_Gen_{i}","book_id":None}
fori,xinenumerate(raw)
]
defbuild_class_c():
    """Load all Class C paragraphs from JSON."""
withopen("data/processed/class_c_dickens.json")asf:
        c_dickens=json.load(f)
withopen("data/processed/class_c_austen.json")asf:
        c_austen=json.load(f)
result=[]
fori,xinenumerate(c_dickens):
        result.append({"text":x["text"],"label_class":"C","label_author":"Dickens",
"source_id":f"AI_Imp_D_{i}","book_id":None})
fori,xinenumerate(c_austen):
        result.append({"text":x["text"],"label_class":"C","label_author":"Austen",
"source_id":f"AI_Imp_A_{i}","book_id":None})
returnresult
defstrict_split(class_a,class_b,class_c,seed=42):
    """
    Split with book-level isolation for Class A.
    Class A: 3 books train, 1 val, 1 test per author.
    Class B/C: 60/20/20 random split.
    """
rng=random.Random(seed)
train,val,test=[],[],[]
forauthor_name,book_idsin[("Dickens",DICKENS_IDS),("Austen",AUSTEN_IDS)]:
        ids=list(book_ids)
rng.shuffle(ids)
train_ids=set(ids[:3])
val_ids=set(ids[3:4])
test_ids=set(ids[4:])
forpinclass_a:
            ifp["label_author"]!=author_name:
                continue
bid=p["book_id"]
ifbidintrain_ids:
                train.append(p)
elifbidinval_ids:
                val.append(p)
elifbidintest_ids:
                test.append(p)
fordata_listin[class_b,class_c]:
        items=list(data_list)
rng.shuffle(items)
n=len(items)
t_end=int(n*0.6)
v_end=int(n*0.8)
train.extend(items[:t_end])
val.extend(items[t_end:v_end])
test.extend(items[v_end:])
returntrain,val,test
deffinalize(data_list,name,seed=42):
    df=pd.DataFrame(data_list)
df["label"]=df["label_class"].map({"A":"Human","B":"AI_Generic","C":"AI_Impostor"})
df["author"]=df["label_author"]
df=df.sample(frac=1,random_state=seed).reset_index(drop=True)
df["sample_id"]=[f"{name}_{i}"foriinrange(len(df))]
if"book_id"indf.columns:
        df=df.drop(columns=["book_id"])
df.to_csv(f"data/processed/{name}.csv",index=False)
returndf
defmain():
    config=load_config()
seed=config["project"]["seed"]
random.seed(seed)
print("="*60)
print("REASSEMBLING DATASET WITH EXPANDED DATA")
print("="*60)
print("\n--- Class A (Human) ---")
class_a=build_class_a(config,target_per_author=TARGET_PER_CLASS//2)
print(f"Total Class A: {len(class_a)}")
print("\n--- Class B (AI Generic) ---")
class_b=build_class_b()
print(f"Total Class B: {len(class_b)}")
print("\n--- Class C (AI Impostor) ---")
class_c=build_class_c()
print(f"Total Class C: {len(class_c)}")
print("\n--- Splitting (book-level isolation for humans) ---")
train_data,val_data,test_data=strict_split(class_a,class_b,class_c,seed=seed)
train_df=finalize(train_data,"train",seed)
val_df=finalize(val_data,"val",seed)
test_df=finalize(test_data,"test",seed)
all_df=pd.concat([train_df,val_df,test_df],ignore_index=True)
all_df.to_csv("data/processed/all.csv",index=False)
print(f"\n  Train: {len(train_df)}")
print(f"  Val:   {len(val_df)}")
print(f"  Test:  {len(test_df)}")
print(f"  Total: {len(all_df)}")
print(f"\n  Label distribution:\n{all_df['label'].value_counts().to_string()}")
print("\n"+"="*60)
print("DATASET ASSEMBLY COMPLETE")
print("="*60)
if__name__=="__main__":
    main()