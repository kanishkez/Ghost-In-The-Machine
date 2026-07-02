"""Shared utilities for Gemini client, text cleaning, and IO."""
from__future__importannotations
importos
importre
importtime
frompathlibimportPath
fromtypingimportList
importpandasaspd
importgoogle.generativeaiasgenai
from.configimportCFG,PROJECT_ROOT
_GUTENBERG_START=re.compile(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*",re.DOTALL)
_GUTENBERG_END=re.compile(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*",re.DOTALL)
classGeminiClient:
    def__init__(self,model_name:str="gemini-2.5-flash"):
        api_key=os.environ.get("GEMINI_API_KEY")
ifnotapi_key:
            raiseEnvironmentError("Set GEMINI_API_KEY env var.")
genai.configure(api_key=api_key)
self.model=genai.GenerativeModel(model_name)
defgenerate(self,prompt:str,max_retries:int=10,**kwargs)->str:
        forattemptinrange(max_retries):
            try:
                time.sleep(12)
resp=self.model.generate_content(prompt,**kwargs)
returnresp.text.strip()
exceptExceptionase:
                match=re.search(r'retry in (\d+(?:\.\d+)?)s',str(e))
wait=float(match.group(1))+5ifmatchelse(2**attempt)+65
print(f"[Gemini retry {attempt+1}/{max_retries}] sleeping {wait:.0f}s",flush=True)
time.sleep(wait)
raiseRuntimeError(f"Gemini failed after {max_retries} retries.")
defstrip_gutenberg_boilerplate(text:str)->str:
    m=_GUTENBERG_START.search(text)
ifm:
        text=text[m.end():]
m=_GUTENBERG_END.search(text)
ifm:
        text=text[:m.start()]
returntext
defclean_text(text:str)->str:
    text=strip_gutenberg_boilerplate(text)
text=re.sub(r"(?im)^(CHAPTER|VOLUME|BOOK|PART)\s+[A-Z0-9IVXLCDM]+\.?\s*$","",text)
text=re.sub(r"(?m)^[A-Z\s.,;:\"'-]{5,}$","",text)
text=re.sub(r"\[.*?\]","",text)
text=re.sub(r"(?im)^[IVXLCDM]+\.?\s*$","",text)
text=re.sub(r"\r\n","\n",text)
text=re.sub(r"\n{3,}","\n\n",text)
text=re.sub(r"[ \t]{2,}"," ",text)
returntext.strip()
defsplit_into_paragraphs(text:str,min_words:int=100,max_words:int=200)->List[str]:
    raw=[p.strip()forpintext.split("\n\n")ifp.strip()]
out=[]
forpinraw:
        wc=len(p.split())
ifmin_words<=wc<=max_words:
            out.append(p)
elifwc>max_words:
            sentences=re.split(r"(?<=[.!?])\s+",p)
buf,buf_wc=[],0
forsinsentences:
                sw=len(s.split())
ifbuf_wc+sw>max_wordsandbuf:
                    ifmin_words<=buf_wc<=max_words:
                        out.append(" ".join(buf))
buf,buf_wc=[s],sw
else:
                    buf.append(s)
buf_wc+=sw
ifbufandmin_words<=buf_wc<=max_words:
                out.append(" ".join(buf))
returnout
defsave_csv(df:pd.DataFrame,name:str)->Path:
    out_dir=PROJECT_ROOT/CFG["paths"]["processed_dir"]
out_dir.mkdir(parents=True,exist_ok=True)
p=out_dir/name
df.to_csv(p,index=False)
returnp
defload_csv(name:str)->pd.DataFrame:
    returnpd.read_csv(PROJECT_ROOT/CFG["paths"]["processed_dir"]/name)