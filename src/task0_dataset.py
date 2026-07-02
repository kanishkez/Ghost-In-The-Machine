"""
Task 0 — The Library of Babel (Two-Author Design).
Class A (Human): 250 Dickens + 250 Austen paragraphs. Topic = None.
Class B (AI-generic): 500 Gemini paragraphs on 10 topics. (Literary Fiction)
Class C (AI-impostor): 250 Gemini imitating Dickens + 250 imitating Austen.
Splits on label + author to prevent train/test contamination.
"""
from__future__importannotations
importrandom
frompathlibimportPath
importurllib.request
importpandasaspd
fromtqdmimporttqdm
from.configimportCFG,PROJECT_ROOT,seed_everything
from.utilsimportGeminiClient,clean_text,split_into_paragraphs,save_csv
defdownload_gutenberg(book_id:int,out_path:Path)->Path:
    url=f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
ifout_path.exists():returnout_path
urllib.request.urlretrieve(url,out_path)
returnout_path
defbuild_class_a()->pd.DataFrame:
    raw_dir=PROJECT_ROOT/CFG["paths"]["raw_dir"]
raw_dir.mkdir(parents=True,exist_ok=True)
n_per_author=CFG["dataset"]["n_per_author"]
all_paragraphs=[]
forauthor,booksinCFG["gutenberg_books"].items():
        author_paragraphs=[]
forbookinbooks:
            path=raw_dir/f"{book['id']}_{book['title'].replace(' ','_')}.txt"
try:
                download_gutenberg(book["id"],path)
exceptExceptionase:
                print(f"[skip] {book['title']}: {e}")
continue
text=clean_text(path.read_text(encoding="utf-8",errors="ignore"))
author_paragraphs.extend(split_into_paragraphs(
text,
min_words=CFG["dataset"]["paragraph_min_words"],
max_words=CFG["dataset"]["paragraph_max_words"]
))
assertlen(author_paragraphs)>=n_per_author,f"Not enough paragraphs for {author}: found {len(author_paragraphs)}, needed {n_per_author}"
random.shuffle(author_paragraphs)
author_paragraphs=author_paragraphs[:n_per_author]
print(f"[Class A] kept {len(author_paragraphs)} {author} paragraphs.")
forpinauthor_paragraphs:
            all_paragraphs.append({"text":p,"label":"Human","author":author,"topic":None})
returnpd.DataFrame(all_paragraphs)
_BUILD_B_PROMPT="""Write a single paragraph of 100-200 words on the topic: "{topic}".
Write in the style of literary fiction. Do not reference any specific author or style.
Return only the paragraph text."""
defbuild_class_b()->pd.DataFrame:
    client=GeminiClient()
topics=CFG["dataset"]["topics"]
rows=[]
n_per_topic=CFG["dataset"]["n_per_class"]//len(topics)
fortintqdm(topics,desc="Class B"):
        for_inrange(n_per_topic):
            txt=client.generate(_BUILD_B_PROMPT.format(topic=t))
rows.append({"text":txt,"label":"AI_Generic","author":None,"topic":t})
returnpd.DataFrame(rows)
defbuild_class_c()->pd.DataFrame:
    client=GeminiClient()
topics=CFG["dataset"]["topics"]
rows=[]
assertCFG["dataset"]["n_per_author"]%len(topics)==0,"n_per_author must be divisible by number of topics"
n_per_topic=CFG["dataset"]["n_per_author"]//len(topics)
style_map={
"Charles Dickens":"Mimic the prose style of Charles Dickens: long, serpentine sentences; rich, sometimes grotesque, imagery; sentimental moral reflection; archaic diction; liberal use of semicolons and em-dashes; concrete London detail.",
"Jane Austen":"Mimic the prose style of Jane Austen: precise, balanced clauses; sharp irony; free indirect discourse; focus on manners and social nuance; elegant, witty vocabulary."
}
forauthorinCFG["project"]["authors"]:
        print(f"Generating Class C for {author}...")
prompt_template=f"""Write a single paragraph of 100-200 words on the topic: "{topic}".
{style_map[author]}
Do NOT name the author or quote them. Return only the paragraph text."""
fortintqdm(topics,desc=f"Class C ({author})"):
            for_inrange(n_per_topic):
                txt=client.generate(prompt_template.format(topic=t))
rows.append({"text":txt,"label":"AI_Impostor","author":author,"topic":t})
returnpd.DataFrame(rows)
defstratified_split(df:pd.DataFrame,ratios=(0.70,0.15,0.15),seed:int=42):
    fromsklearn.model_selectionimporttrain_test_split
df["strat_key"]=df["label"]+"_"+df["author"].fillna("Generic")
train_df,temp_df=train_test_split(df,test_size=1-ratios[0],stratify=df["strat_key"],random_state=seed)
val_df,test_df=train_test_split(temp_df,test_size=ratios[2]/(ratios[1]+ratios[2]),stratify=temp_df["strat_key"],random_state=seed)
returntrain_df.reset_index(drop=True),val_df.reset_index(drop=True),test_df.reset_index(drop=True)
defmain():
    seed_everything(CFG["project"]["seed"])
print("=== Building Class A (Human) ===")
df_a=build_class_a()
print("=== Building Class B (AI generic) ===")
df_b=build_class_b()
print("=== Building Class C (AI impostor) ===")
df_c=build_class_c()
df=pd.concat([df_a,df_b,df_c],ignore_index=True)
df=df.sample(frac=1,random_state=42).reset_index(drop=True)
df["sample_id"]=range(len(df))
print(f"Total samples: {len(df)}\n{df['label'].value_counts()}")
train,val,test=stratified_split(df,ratios=(CFG["split"]["train"],CFG["split"]["val"],CFG["split"]["test"]))
save_csv(train,"train.csv");save_csv(val,"val.csv");save_csv(test,"test.csv")
save_csv(df,"all.csv")
print(f"train={len(train)} val={len(val)} test={len(test)}")
if__name__=="__main__":
    main()