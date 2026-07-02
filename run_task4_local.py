"""
Task 4 — The Turing Test (Local Version).
Genetic algorithm that evolves AI-generated text to fool the Tier C detector.
Uses local text perturbation instead of Gemini API for mutation.
"""
importjson
importrandom
importre
importsys
frompathlibimportPath
fromtypingimportList
importnumpyasnp
importtorch
fromtransformersimportAutoTokenizer,AutoModelForSequenceClassification
frompeftimportPeftModel
sys.path.insert(0,str(Path(__file__).parent.parent))
fromsrc.configimportCFG,PROJECT_ROOT,seed_everything
fromsrc.utilsimportload_csv
classHumanProbOracle:
    def__init__(self,device="cpu"):
        self.device=device
base=AutoModelForSequenceClassification.from_pretrained(
CFG["tier_c"]["base_model"],num_labels=3
)
self.model=PeftModel.from_pretrained(
base,PROJECT_ROOT/CFG["paths"]["models_dir"]/"tier_c_lora"
).to(device).eval()
self.tok=AutoTokenizer.from_pretrained(
PROJECT_ROOT/CFG["paths"]["models_dir"]/"tier_c_lora"
)
@torch.no_grad()
defpredict_proba(self,texts:List[str])->np.ndarray:
        enc=self.tok(
texts,return_tensors="pt",truncation=True,
padding=True,max_length=CFG["tier_c"]["max_length"]
).to(self.device)
logits=self.model(**enc).logits.cpu().numpy()
e=np.exp(logits-logits.max(axis=1,keepdims=True))
probs=e/e.sum(axis=1,keepdims=True)
returnprobs[:,0]
ARCHAIC_WORDS={
"very":["exceedingly","uncommonly","tolerably"],
"said":["observed","remarked","declared","pronounced"],
"big":["considerable","vast","prodigious"],
"small":["diminutive","trifling","inconsiderable"],
"happy":["contented","gratified","well-pleased"],
"sad":["melancholy","sorrowful","dejected"],
"good":["agreeable","estimable","admirable"],
"bad":["disagreeable","wretched","lamentable"],
"old":["aged","venerable","ancient"],
"new":["novel","fresh","recent"],
"think":["apprehend","conceive","fancy"],
"know":["perceive","discern","comprehend"],
"want":["desire","wish","require"],
"like":["favour","esteem","regard"],
"walk":["proceed","advance","make one's way"],
"look":["regard","survey","contemplate"],
"house":["dwelling","residence","establishment"],
"people":["persons","souls","individuals"],
"street":["thoroughfare","lane","passage"],
"money":["fortune","means","pecuniary resources"],
"however":["nevertheless","notwithstanding","withal"],
"but":["yet","however","nevertheless"],
"also":["moreover","furthermore","likewise"],
"about":["concerning","touching upon","regarding"],
"because":["for","inasmuch as","owing to the circumstance that"],
}
SENTENCE_STARTERS=[
"It must be observed that ","Be it known that ",
"One might venture to say that ","It chanced that ",
"There was, in truth, ","It would not be too much to say that ",
"Let it be recorded that ","It was a circumstance of some note that ",
]
INTERJECTIONS=[
" — if such a word may be permitted — ",
" (a matter of no small consequence) ",
", it must be confessed, ",
", to speak plainly, ",
" — for so it appeared — ",
", one might say, ",
]
defmutate_swap_archaic(text:str)->str:
    """Replace common words with archaic/literary equivalents."""
words=text.split()
fori,winenumerate(words):
        clean=w.strip(".,;:!?\"'()").lower()
ifcleaninARCHAIC_WORDSandrandom.random()<0.3:
            replacement=random.choice(ARCHAIC_WORDS[clean])
ifw[0].isupper():
                replacement=replacement.capitalize()
trailing=""
forcinreversed(w):
                ifcin".,;:!?\"'()":
                    trailing=c+trailing
else:
                    break
words[i]=replacement+trailing
return" ".join(words)
defmutate_vary_rhythm(text:str)->str:
    """Split or merge sentences to change rhythm."""
sentences=re.split(r'(?<=[.!?])\s+',text)
iflen(sentences)<2:
        returntext
result=[]
i=0
whilei<len(sentences):
        ifrandom.random()<0.25andi+1<len(sentences):
            merged=sentences[i].rstrip(".")+"; "+sentences[i+1][0].lower()+sentences[i+1][1:]
result.append(merged)
i+=2
elifrandom.random()<0.2and", "insentences[i]:
            parts=sentences[i].split(", ",1)
result.append(parts[0]+".")
result.append(parts[1].capitalize())
i+=1
else:
            result.append(sentences[i])
i+=1
return" ".join(result)
defmutate_add_interjection(text:str)->str:
    """Insert a parenthetical interjection into a random sentence."""
sentences=re.split(r'(?<=[.!?])\s+',text)
ifnotsentences:
        returntext
idx=random.randint(0,len(sentences)-1)
s=sentences[idx]
words=s.split()
iflen(words)>6:
        insert_pos=random.randint(3,len(words)-3)
interjection=random.choice(INTERJECTIONS)
words.insert(insert_pos,interjection.strip())
sentences[idx]=" ".join(words)
return" ".join(sentences)
defmutate_add_starter(text:str)->str:
    """Prepend a literary sentence starter."""
sentences=re.split(r'(?<=[.!?])\s+',text)
ifsentences:
        idx=random.randint(0,min(1,len(sentences)-1))
starter=random.choice(SENTENCE_STARTERS)
sentences[idx]=starter+sentences[idx][0].lower()+sentences[idx][1:]
return" ".join(sentences)
defmutate_shuffle_clauses(text:str)->str:
    """Reverse the order of clauses within a random sentence."""
sentences=re.split(r'(?<=[.!?])\s+',text)
foriinrange(len(sentences)):
        if", "insentences[i]andrandom.random()<0.3:
            clauses=sentences[i].split(", ")
iflen(clauses)>=2:
                random.shuffle(clauses)
sentences[i]=", ".join(clauses)
return" ".join(sentences)
MUTATORS=[
mutate_swap_archaic,
mutate_vary_rhythm,
mutate_add_interjection,
mutate_add_starter,
mutate_shuffle_clauses,
]
defmutate(paragraph:str)->str:
    """Apply 1-2 random mutations."""
n_mutations=random.randint(1,2)
for_inrange(n_mutations):
        fn=random.choice(MUTATORS)
paragraph=fn(paragraph)
returnparagraph
defcrossover(p1:str,p2:str)->str:
    """Interleave sentences from two parents."""
s1=re.split(r'(?<=[.!?])\s+',p1)
s2=re.split(r'(?<=[.!?])\s+',p2)
result=[]
fora,binzip(s1,s2):
        result.append(aifrandom.random()<0.5elseb)
longer=s1iflen(s1)>len(s2)elses2
result.extend(longer[len(result):])
return" ".join(result)
defrun_ga():
    seed_everything(CFG["project"]["seed"])
log_dir=PROJECT_ROOT/CFG["paths"]["ga_logs"]/"dickens"
log_dir.mkdir(parents=True,exist_ok=True)
device="mps"iftorch.backends.mps.is_available()else"cpu"
oracle=HumanProbOracle(device=device)
test=load_csv("test.csv")
impostors=test[test["label"]=="AI_Impostor"]["text"].tolist()
random.shuffle(impostors)
pop_size=CFG["genetic_algorithm"]["population_size"]
pop=impostors[:pop_size]
print(f"=== GA: Evolving {pop_size} paragraphs to fool the detector ===\n")
log=[]
generations=CFG["genetic_algorithm"]["generations"]
elite_size=CFG["genetic_algorithm"]["elite_size"]
target=CFG["genetic_algorithm"]["target_human_prob"]
forgeninrange(generations):
        probs=oracle.predict_proba(pop)
ranked=sorted(zip(pop,probs),key=lambdax:-x[1])
best=ranked[0][1]
mean=float(np.mean(probs))
print(f"[Gen {gen}] best P(Human)={best:.4f}  mean={mean:.4f}")
log.append({
"generation":gen,
"best":float(best),
"mean":mean,
"best_text":ranked[0][0][:300],
})
ifbest>=target:
            print(f"  ✓ Reached target >= {target}!")
break
elites=[tfort,_inranked[:elite_size]]
new_pop=list(elites)
whilelen(new_pop)<pop_size:
            ifrandom.random()<0.3andlen(elites)>=2:
                p1,p2=random.sample(elites,2)
child=crossover(p1,p2)
else:
                parent=random.choice(elites)
child=mutate(parent)
new_pop.append(child)
pop=new_pop
probs=oracle.predict_proba(pop)
ranked=sorted(zip(pop,probs),key=lambdax:-x[1])
print(f"\n=== Final best P(Human) = {ranked[0][1]:.4f} ===")
print(f"\nBest paragraph:\n{ranked[0][0][:500]}")
withopen(log_dir/"ga_log.json","w")asf:
        json.dump(log,f,indent=2)
withopen(log_dir/"ga_best.txt","w")asf:
        f.write(ranked[0][0])
print(f"\nLogs saved to {log_dir}")
returnranked
defpersonal_test():
    """Run the detector on the user's own writing if available."""
sop_path=PROJECT_ROOT/"data"/"processed"/"full_user_paste.txt"
ifnotsop_path.exists():
        print("\n[Personal Test] No SOP/essay found at data/processed/full_user_paste.txt — skipping.")
return
device="mps"iftorch.backends.mps.is_available()else"cpu"
oracle=HumanProbOracle(device=device)
text=sop_path.read_text(encoding="utf-8")
paragraphs=[p.strip()forpintext.split("\n\n")iflen(p.split())>=50]
ifnotparagraphs:
        print("[Personal Test] No paragraphs with >= 50 words found.")
return
print(f"\n=== Personal Test: {len(paragraphs)} paragraphs ===\n")
probs=oracle.predict_proba(paragraphs)
fori,(p,prob)inenumerate(zip(paragraphs,probs)):
        verdict="HUMAN ✓"ifprob>0.5else"AI ✗"
print(f"[{i+1}] P(Human)={prob:.3f}  → {verdict}")
print(f"    {p[:150]}...\n")
human_count=sum(1forpinprobsifp>0.5)
print(f"Summary: {human_count}/{len(paragraphs)} classified as Human")
if__name__=="__main__":
    ranked=run_ga()
personal_test()