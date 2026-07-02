"""
Generate ~140 additional paragraphs per class to scale up the dataset.
Class A (Human): Extract more from already-downloaded Gutenberg books.
Class B (AI Generic): Generate via Gemini on the same 10 topics.
Class C (AI Impostor): Generate via Gemini mimicking Dickens/Austen.
After generation, reassemble the dataset and re-run the full pipeline.
"""
importjson
importos
importsys
importglob
importrandom
importtime
frompathlibimportPath
TARGET_PER_CLASS=240
ADDITIONAL_NEEDED=140
TOPICS=[
"Poverty and the Condition of the Working Class",
"The Innocence and Suffering of Children",
"Crime, Guilt, and the Justice System",
"The Hypocrisy of Society",
"Family, Loyalty, and Betrayal",
"The City as a Living Organism",
"Class Mobility and Social Aspiration",
"Memory, Regret, and Redemption",
"Love and Sacrifice",
"The Comedic and the Grotesque",
]
DICKENS_IDS=["98","1400","730","766","46"]
AUSTEN_IDS=["105","121","1342","158","161"]
CLASS_B_PROMPT="""Write a single original paragraph of 100-200 words on the following topic:
"{topic}"
Requirements:
- Write in the style of contemporary literary fiction.
- Do NOT reference any specific author, style, or literary tradition.
- Do NOT use meta-commentary like "In conclusion" or "This paragraph explores."
- Use vivid, concrete details rather than abstract generalizations.
- Vary your sentence structure naturally.
- Return ONLY the paragraph text with no preamble or commentary."""
CLASS_C_DICKENS_PROMPT="""Write a single original paragraph of 100-200 words on the following topic:
"{topic}"
You must write in the exact prose style of Charles Dickens. Specifically:
- Use long, serpentine sentences with multiple embedded clauses joined by semicolons and em-dashes.
- Employ rich, sometimes grotesque or exaggerated imagery.
- Include sentimental moral reflection and emotional commentary.
- Use slightly archaic Victorian diction ("whereupon," "it chanced that," "be it observed").
- Add concrete details about London: fog, gaslight, cobblestones, the Thames, crowded streets.
- Blend dark humor with pathos.
Do NOT name Charles Dickens or quote him directly.
Return ONLY the paragraph text with no preamble."""
CLASS_C_AUSTEN_PROMPT="""Write a single original paragraph of 100-200 words on the following topic:
"{topic}"
You must write in the exact prose style of Jane Austen. Specifically:
- Use precise, balanced clauses with careful syntactic parallelism.
- Employ sharp irony and understated wit — say the opposite of what you mean.
- Use free indirect discourse (blend the narrator's voice with a character's thoughts).
- Focus on manners, propriety, social nuance, and domestic life.
- Use elegant, measured vocabulary appropriate to Regency-era England.
- Let observations about human nature emerge through social situations, not declarations.
Do NOT name Jane Austen or quote her directly.
Return ONLY the paragraph text with no preamble."""
defsetup_gemini():
    """Initialize Gemini client."""
importgoogle.generativeaiasgenai
api_key=os.environ.get("GEMINI_API_KEY")
ifnotapi_key:
        print("ERROR: Set GEMINI_API_KEY environment variable.")
sys.exit(1)
genai.configure(api_key=api_key)
returngenai.GenerativeModel("gemini-2.5-flash")
defgenerate_with_retry(model,prompt,max_retries=10):
    """Generate text with exponential backoff."""
importre
forattemptinrange(max_retries):
        try:
            time.sleep(5)
resp=model.generate_content(prompt)
text=resp.text.strip()
text=re.sub(r'^```[\w]*\n?','',text)
text=re.sub(r'\n?```$','',text)
returntext.strip()
exceptExceptionase:
            err=str(e)
match=re.search(r'retry in (\d+(?:\.\d+)?)s',err)
wait=float(match.group(1))+5ifmatchelse(2**attempt)+30
print(f"  [retry {attempt+1}/{max_retries}] waiting {wait:.0f}s — {err[:80]}")
time.sleep(wait)
raiseRuntimeError(f"Gemini failed after {max_retries} retries")
defextract_class_a(needed_per_author=70):
    """Extract additional human paragraphs from existing Gutenberg books."""
sys.path.insert(0,str(Path(__file__).parent))
fromsrc.utilsimportclean_text,split_into_paragraphs
raw_dir=Path("data/raw")
results=[]
forauthor,idsin[("Dickens",DICKENS_IDS),("Austen",AUSTEN_IDS)]:
        pool=[]
forfpathinsorted(raw_dir.glob("*.txt")):
            ifany(fpath.stem.startswith(f"{bid}_")forbidinids):
                text=clean_text(fpath.read_text(encoding="utf-8",errors="ignore"))
paras=split_into_paragraphs(text,100,200)
forpinparas:
                    pool.append({"text":p,"book":fpath.stem,"author":author})
random.shuffle(pool)
selected=pool[:needed_per_author+50]
results.extend(selected[:needed_per_author])
print(f"  [Class A] {author}: selected {min(needed_per_author,len(selected))} from {len(pool)} available")
returnresults
defgenerate_class_b(model,count=140):
    """Generate additional generic AI paragraphs."""
per_topic=count//len(TOPICS)
remainder=count%len(TOPICS)
results=[]
fori,topicinenumerate(TOPICS):
        n=per_topic+(1ifi<remainderelse0)
forjinrange(n):
            prompt=CLASS_B_PROMPT.format(topic=topic)
text=generate_with_retry(model,prompt)
results.append({"text":text,"topic":topic})
total_done=len(results)
print(f"  [Class B] {total_done}/{count} — topic: {topic[:40]}...")
returnresults
defgenerate_class_c(model,count_per_author=70):
    """Generate additional impostor paragraphs for both authors."""
results_dickens=[]
results_austen=[]
per_topic=count_per_author//len(TOPICS)
remainder=count_per_author%len(TOPICS)
print(f"\n  Generating {count_per_author} Dickens impostors...")
fori,topicinenumerate(TOPICS):
        n=per_topic+(1ifi<remainderelse0)
forjinrange(n):
            prompt=CLASS_C_DICKENS_PROMPT.format(topic=topic)
text=generate_with_retry(model,prompt)
results_dickens.append({"text":text,"topic":topic})
print(f"  [Class C / Dickens] {len(results_dickens)}/{count_per_author}")
print(f"\n  Generating {count_per_author} Austen impostors...")
fori,topicinenumerate(TOPICS):
        n=per_topic+(1ifi<remainderelse0)
forjinrange(n):
            prompt=CLASS_C_AUSTEN_PROMPT.format(topic=topic)
text=generate_with_retry(model,prompt)
results_austen.append({"text":text,"topic":topic})
print(f"  [Class C / Austen] {len(results_austen)}/{count_per_author}")
returnresults_dickens,results_austen
defmain():
    random.seed(42)
processed=Path("data/processed")
withopen(processed/"class_b.json")asf:
        existing_b=json.load(f)
withopen(processed/"class_c_dickens.json")asf:
        existing_c_d=json.load(f)
withopen(processed/"class_c_austen.json")asf:
        existing_c_a=json.load(f)
print(f"Existing data: B={len(existing_b)}, C_Dickens={len(existing_c_d)}, C_Austen={len(existing_c_a)}")
needed_b=max(0,TARGET_PER_CLASS-len(existing_b))
needed_c_per_author=max(0,(TARGET_PER_CLASS//2)-len(existing_c_d))
print(f"Need to generate: B={needed_b}, C_per_author={needed_c_per_author}")
print("\n=== Step 1: Extracting more Class A (Human) paragraphs ===")
class_a_extra=extract_class_a(needed_per_author=TARGET_PER_CLASS//2)
withopen(processed/"class_a_extra.json","w")asf:
        json.dump(class_a_extra,f,indent=2)
print(f"  Saved {len(class_a_extra)} extra Class A paragraphs")
ifneeded_b>0:
        print(f"\n=== Step 2: Generating {needed_b} more Class B paragraphs ===")
model=setup_gemini()
new_b=generate_class_b(model,count=needed_b)
all_b=existing_b+new_b
withopen(processed/"class_b.json","w")asf:
            json.dump(all_b,f,indent=2)
print(f"  Total Class B: {len(all_b)}")
else:
        print(f"\n=== Step 2: Class B already has {len(existing_b)} — skipping ===")
model=setup_gemini()
ifneeded_c_per_author>0:
        print(f"\n=== Step 3: Generating {needed_c_per_author} more Class C per author ===")
new_c_d,new_c_a=generate_class_c(model,count_per_author=needed_c_per_author)
all_c_d=existing_c_d+new_c_d
all_c_a=existing_c_a+new_c_a
withopen(processed/"class_c_dickens.json","w")asf:
            json.dump(all_c_d,f,indent=2)
withopen(processed/"class_c_austen.json","w")asf:
            json.dump(all_c_a,f,indent=2)
print(f"  Total Class C: Dickens={len(all_c_d)}, Austen={len(all_c_a)}")
else:
        print(f"\n=== Step 3: Class C already sufficient — skipping ===")
print("\n=== Data generation complete! ===")
print("Run the assembly + pipeline next:")
print("  python3 reassemble_and_run.py")
if__name__=="__main__":
    main()